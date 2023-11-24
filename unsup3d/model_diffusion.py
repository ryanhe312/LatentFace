import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from . import networks
from . import utils
from .renderer_pytorch3d import Renderer

import lpips
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from diffusers import UNet2DModel, DDIMScheduler

EPS = 1e-7


class Unsup3D_diffusion():
    def __init__(self, cfgs):
        super().__init__()
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.renderer = Renderer(cfgs)
        self.lam_reg = cfgs.get('lam_reg', 0.1)
        self.weight_depth = cfgs.get('weight_depth', 1)
        self.weight_albedo = cfgs.get('weight_albedo', 0.1)
        self.run_test = cfgs.get('run_test', False)

        ## networks and optimizers
        repeat = int(math.log2(self.image_size / 64))
        self.netEA = UNet2DModel(
            sample_size=1,
            in_channels=512,
            out_channels=256,
            layers_per_block=2,
            block_out_channels=(1024,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
        )
        self.netED = UNet2DModel(
            sample_size=1,
            in_channels=512,
            out_channels=256,
            layers_per_block=2,
            block_out_channels=(1024,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
        )
        self.scheduler = DDIMScheduler(prediction_type='sample')
        self.scheduler.set_timesteps(5)
        self.netD = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None, repeat=repeat).requires_grad_(False)
        self.netA = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256, repeat=repeat).requires_grad_(False)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32, repeat=repeat).requires_grad_(False)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32, repeat=repeat).requires_grad_(False)
        self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128, repeat=repeat).requires_grad_(False)
        self.network_names = [k for k in vars(self) if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.make_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5)

        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth

    def init_optimizers(self):
        self.optimizer_names = []
        self.scheduler_names = []
        for net_name in ['netEA','netED']:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
            scheduler = self.make_scheduler(optimizer)
            scheduler_name = net_name.replace('net','scheduler')
            setattr(self, scheduler_name, scheduler)
            self.scheduler_names += [scheduler_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def distribution_loss(self, input, dim, var):
        mu = input.mean(dim=dim)
        sigma = input.var(dim=dim)
        p1 = torch.mean(torch.pow(mu, 2.0))
        p2 = torch.mean(torch.pow(torch.pow(sigma, 1 / 2) - var, 2.0))
        return p1 + p2

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.netEA.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netED.parameters(), 1.0)
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()
        for scheduler_name in self.scheduler_names:
            getattr(self, scheduler_name).step()

    def extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        res = arr[timesteps].float().to(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def diffusion_forward(self, code, net):
        clean_images = code.mean(0).expand_as(code)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bsz = clean_images.shape[0]
        timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        concat_input = torch.cat([noisy_images, code], dim=1)
        model_output = net(concat_input, timesteps).sample
        # self.loss_diffusion = F.mse_loss(model_output, noise)

        # predict sample not epsilon
        alpha_t = self.extract_into_tensor(
            self.scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
        )
        snr_weights = alpha_t / (1 - alpha_t)
        loss_diffusion = snr_weights * F.mse_loss(
            model_output, clean_images, reduction="none"
        )  # use SNR weighting from distillation paper
        return loss_diffusion, model_output

    def depth_process(self, depth_raw):
        b = depth_raw.shape[0]
        depth = depth_raw - depth_raw.view(b,-1).mean(1).view(b,1,1)
        depth = depth.tanh()
        depth = self.depth_rescaler(depth)
        return depth

    def forward(self, input):
        """Feedforward once."""
        if self.load_gt_depth:
            input, depth_gt = input
        self.input_im = input.to(self.device) *2.-1.
        b, c, h, w = self.input_im.shape

        ## predict canonical depth
        self.depth_code, self.canon_depth_raw = self.netD(self.input_im)  # BxHxW
        self.canon_depth_raw = self.canon_depth_raw.squeeze(1)
        self.canon_depth = self.depth_process(self.canon_depth_raw)

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.albedo_code, self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW
        # self.canon_albedo = self.canon_albedo + self.weight_albedo * self.canon_expr
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

        ## predict confidence map
        self.conf_sigma_l1, self.conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW

        ## predict lighting
        canon_light = self.netL(self.input_im)[0:1].repeat(b,1).repeat(2,1)  # Bx4
        self.canon_light_a = canon_light[:,:1] /2+0.5  # ambience term
        self.canon_light_b = canon_light[:,1:2] /2+0.5  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im).repeat(2,1)
        self.view = torch.cat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_im, self.recon_depth = self.renderer.render(self.canon_depth, self.canon_im, get_depth=True)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        self.recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        self.recon_im_mask_both = self.recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        self.recon_im = self.recon_im * self.recon_im_mask_both

        self.recon_im_with_input = self.recon_im * self.recon_im_mask_both + self.input_im.repeat(2, 1, 1, 1) * (1 - self.recon_im_mask_both)


        ## loss function
        self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=self.recon_im_mask_both[:b], conf_sigma=self.conf_sigma_l1[:,:1])
        self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=self.recon_im_mask_both[b:], conf_sigma=self.conf_sigma_l1[:,1:])
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=self.recon_im_mask_both[:b], conf_sigma=self.conf_sigma_percl[:,:1])
        self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:], self.input_im, mask=self.recon_im_mask_both[b:], conf_sigma=self.conf_sigma_percl[:,1:])
        lam_flip = 0.5 # if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        self.loss_recon = self.loss_l1_im + lam_flip*self.loss_l1_im_flip + self.lam_perc*(self.loss_perc_im + lam_flip*self.loss_perc_im_flip)

        ## Diffusion
        self.loss_diffusion_a, self.output_diffusion_a = self.diffusion_forward(self.albedo_code, self.netEA)
        self.loss_diffusion_d, self.output_diffusion_d = self.diffusion_forward(self.depth_code, self.netED)

        ## total loss
        self.loss_total = self.loss_diffusion_a.mean() + self.loss_diffusion_d.mean()
        metrics = {'loss_recon': self.loss_recon, 'loss_total': self.loss_total}

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        ## render rotations
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(b0,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b0].permute(0,3,1,2), self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)

        input_im = self.input_im[:b0].detach().cpu() /2+0.5
        # input_im_symline = self.input_im_symline[:b0].detach().cpu() /2.+0.5
        canon_albedo = self.canon_albedo[:b0].detach().cpu() /2.+0.5
        canon_im = self.canon_im[:b0].detach().cpu() /2.+0.5
        recon_im = self.recon_im[:b0].detach().cpu() /2.+0.5
        recon_im_flip = self.recon_im[b:b+b0].detach().cpu() /2.+0.5
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1).cpu() /2.+0.5
        canon_depth = ((self.canon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        recon_depth = ((self.recon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        canon_diffuse_shading = self.canon_diffuse_shading[:b0].detach().cpu()
        canon_normal = self.canon_normal.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        recon_normal = self.recon_normal.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        conf_map_l1 = 1/(1+self.conf_sigma_l1[:b0,:1].detach().cpu()+EPS)
        conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b0,1:].detach().cpu()+EPS)
        conf_map_percl = 1/(1+self.conf_sigma_percl[:b0,:1].detach().cpu()+EPS)
        conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b0,1:].detach().cpu()+EPS)

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        ## Diffusion
        average_albedo = self.netA.network2(self.albedo_code.mean(0).expand_as(self.albedo_code))
        predict_albedo = self.netA.network2(self.output_diffusion_a)
        average_albedo = average_albedo[:b0].detach().cpu() /2.+0.5
        predict_albedo = predict_albedo[:b0].detach().cpu() /2.+0.5

        average_depth = self.depth_process(self.netD.network2(self.depth_code.mean(0).expand_as(self.depth_code)).squeeze(1))
        predict_depth = self.depth_process(self.netD.network2(self.output_diffusion_d).squeeze(1))
        average_depth = ((average_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        predict_depth = ((predict_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)

        ## write summary
        logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        logger.add_scalar('Loss/loss_recon', self.loss_recon, total_iter)
        logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im, total_iter)
        logger.add_scalar('Loss/loss_l1_im_flip', self.loss_l1_im_flip, total_iter)
        logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im, total_iter)
        logger.add_scalar('Loss/loss_perc_im_flip', self.loss_perc_im_flip, total_iter)

        logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist, total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/'+vlist[i], self.view[:,i], total_iter)
        logger.add_histogram('Light/canon_light_a', self.canon_light_a, total_iter)
        logger.add_histogram('Light/canon_light_b', self.canon_light_b, total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/'+llist[i], self.canon_light_d[:,i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0**0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        # log_grid_image('Image/input_image_symline', input_im_symline)
        log_grid_image('Image/input_image', input_im)
        log_grid_image('Image/canonical_albedo', canon_albedo)
        log_grid_image('Image/canonical_image', canon_im)
        log_grid_image('Image/recon_image', recon_im)
        log_grid_image('Image/recon_image_flip', recon_im_flip)
        log_grid_image('Image/recon_side', canon_im_rotate[:,0,:,:,:])
        log_grid_image('Image/average_albedo', average_albedo)
        log_grid_image('Image/predict_albedo', predict_albedo)

        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw)
        log_grid_image('Depth/canonical_depth', canon_depth)
        log_grid_image('Depth/recon_depth', recon_depth)
        log_grid_image('Depth/canonical_diffuse_shading', canon_diffuse_shading)
        log_grid_image('Depth/canonical_normal', canon_normal)
        log_grid_image('Depth/recon_normal', recon_normal)
        log_grid_image('Depth/average_depth', average_depth)
        log_grid_image('Depth/predict_depth', predict_depth)

        logger.add_histogram('Image/canonical_albedo_hist', canon_albedo, total_iter)
        logger.add_histogram('Image/canonical_diffuse_shading_hist', canon_diffuse_shading, total_iter)

        log_grid_image('Conf/conf_map_l1', conf_map_l1)
        logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1[:,:1], total_iter)
        log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
        logger.add_histogram('Conf/conf_sigma_l1_flip_hist', self.conf_sigma_l1[:,1:], total_iter)
        log_grid_image('Conf/conf_map_percl', conf_map_percl)
        logger.add_histogram('Conf/conf_sigma_percl_hist', self.conf_sigma_percl[:,:1], total_iter)
        log_grid_image('Conf/conf_map_percl_flip', conf_map_percl_flip)
        logger.add_histogram('Conf/conf_sigma_percl_flip_hist', self.conf_sigma_percl[:,1:], total_iter)

        # logger.add_video('Image_rotate/recon_rotate', canon_im_rotate_grid, total_iter, fps=4)
        # logger.add_video('Image_rotate/canon_normal_rotate', canon_normal_rotate_grid, total_iter, fps=4)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() *1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() /100

            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        # with torch.no_grad():
        #     v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(b,1)
        #     canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
        #     canon_im_rotate = canon_im_rotate.clamp(-1,1).detach().cpu() /2+0.5
        #     canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b].permute(0,3,1,2), self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
        #     canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

        input_im = self.input_im[:b].detach().cpu().numpy() /2+0.5
        # input_im_symline = self.input_im_symline.detach().cpu().numpy() /2.+0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() /2+0.5
        # canon_im = self.canon_im[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        recon_im = self.recon_im_with_input[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        # recon_im_flip = self.recon_im[b:].clamp(-1,1).detach().cpu().numpy() /2+0.5
        canon_depth = ((self.canon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        # recon_depth = ((self.recon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        canon_diffuse_shading = self.canon_diffuse_shading[:b].detach().cpu().numpy()
        # canon_normal = self.canon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        # recon_normal = self.recon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        # conf_map_l1 = 1/(1+self.conf_sigma_l1[:b,:1].detach().cpu().numpy()+EPS)
        # conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b,1:].detach().cpu().numpy()+EPS)
        # conf_map_percl = 1/(1+self.conf_sigma_percl[:b,:1].detach().cpu().numpy()+EPS)
        # conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b,1:].detach().cpu().numpy()+EPS)
        # canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[:b].detach().cpu().numpy()
        # view = self.view[:b].detach().cpu().numpy()

        # canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_im_rotate,1)]  # [(C,H,W)]*T
        # canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        # canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
        # canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        # Diffusion
        neutral_a = torch.randn(self.albedo_code.shape).to(self.albedo_code.device)
        for t in self.scheduler.timesteps:
            concat_input = torch.cat([neutral_a, self.albedo_code], dim=1)
            model_output = self.netEA(concat_input, t).sample
            neutral_a = self.scheduler.step(model_output, t, neutral_a, eta=0).prev_sample

        neutral_d = torch.randn(self.depth_code.shape).to(self.depth_code.device)
        for t in self.scheduler.timesteps:
            concat_input = torch.cat([neutral_d, self.depth_code], dim=1)
            model_output = self.netED(concat_input, t).sample
            neutral_d = self.scheduler.step(model_output, t, neutral_d, eta=0).prev_sample

        ## albedo
        average_albedo_code = self.albedo_code.mean(0).expand_as(self.albedo_code)
        average_albedo = self.netA.network2(average_albedo_code)
        predict_albedo = self.netA.network2(neutral_a)
        average_albedo_numpy = average_albedo[:b].detach().cpu().numpy() /2.+0.5
        predict_albedo_numpy = predict_albedo[:b].detach().cpu().numpy() /2.+0.5

        ## depth
        average_depth_code = self.depth_code.mean(0).expand_as(self.depth_code)
        average_depth = self.depth_process(self.netD.network2(average_depth_code).squeeze(1))
        predict_depth = self.depth_process(self.netD.network2(neutral_d).squeeze(1))

        depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        average_depth = average_depth*(1-depth_border) + depth_border *self.border_depth
        predict_depth = predict_depth*(1-depth_border) + depth_border *self.border_depth

        average_normal = self.renderer.get_normal_from_depth(average_depth)
        average_shading = (average_normal * self.canon_light_d.view(-1,1,1,3)[:b]).sum(3).clamp(min=0).unsqueeze(1)
        average_shading_numpy = average_shading[:b].detach().cpu().numpy()
        predict_normal = self.renderer.get_normal_from_depth(predict_depth)
        predict_shading = (predict_normal * self.canon_light_d.view(-1,1,1,3)[:b]).sum(3).clamp(min=0).unsqueeze(1)
        predict_shading_numpy = predict_shading[:b].detach().cpu().numpy()

        predict_canon_shading = self.canon_light_a[:b].view(-1,1,1,1) + self.canon_light_b[:b].view(-1,1,1,1)*predict_shading
        predict_canon_im = (predict_albedo/2+0.5) * predict_canon_shading *2-1
        self.renderer.set_transform_matrices(self.view[:b])
        predict_recon_im, recon_depth = self.renderer.render(predict_depth, predict_canon_im, get_depth=True)
        recon_im_mask = (recon_depth < self.max_depth+(self.max_depth - self.min_depth) /2).float().unsqueeze(1)
        predict_recon_im_with_input = predict_recon_im * recon_im_mask[:b] + self.input_im * (1 - recon_im_mask[:b])
        predict_recon_im_numpy = predict_recon_im_with_input[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        # utils.save_images(save_dir, input_im_symline, suffix='input_image_symline', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        # utils.save_images(save_dir, canon_im, suffix='canonical_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        # utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        # utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
        # utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
        # utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
        # utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
        # utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
        # utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
        # utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)
        # utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        # utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        # utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
        # utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)

        # Diffusion
        utils.save_images(save_dir, average_albedo_numpy, suffix='average_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, predict_albedo_numpy, suffix='predict_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, average_shading_numpy, suffix='average_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, predict_shading_numpy, suffix='predict_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, predict_recon_im_numpy, suffix='predict_recon_im', sep_folder=sep_folder)

        if not hasattr(self, 'lpips'):
            self.lpips = lpips.LPIPS(net='alex').to(self.input_im.device)

        psnr = []
        ssim = []
        lpips_score = self.lpips(self.input_im[:b],self.recon_im_with_input[:b]).squeeze().cpu().numpy()

        imgs = ((self.input_im[:b] + 1)/2).transpose(1, 2).transpose(2, 3).cpu().numpy()
        input = ((self.recon_im_with_input[:b] + 1)/2).transpose(1, 2).transpose(2, 3).cpu().numpy()

        for x, y in zip(input, imgs):
            psnr.append(peak_signal_noise_ratio(x, y, data_range=1))
            ssim.append(structural_similarity(x, y, data_range=1, multichannel=True))

        origin_albedo_distance = nn.functional.cosine_similarity(self.albedo_code, average_albedo_code, dim=1).squeeze().cpu().numpy()
        predict_albedo_distance = nn.functional.cosine_similarity(neutral_a, average_albedo_code, dim=1).squeeze().cpu().numpy()
        origin_shape_distance = nn.functional.cosine_similarity(self.depth_code, average_depth_code, dim=1).squeeze().cpu().numpy()
        predict_shape_distance = nn.functional.cosine_similarity(neutral_d, average_depth_code, dim=1).squeeze().cpu().numpy()

        all_scores = np.stack([psnr,ssim,lpips_score, origin_albedo_distance, predict_albedo_distance, origin_shape_distance, predict_shape_distance],axis=1)

        if not hasattr(self, 'all_scores'):
            self.all_scores = all_scores
        else:
            self.all_scores = np.concatenate([self.all_scores,all_scores],axis=0)

    def save_scores(self, path):
        # save scores if gt is loaded
        header = 'PSNR_V, SSIM_V, LPIPS_V, Origin Albedo Distance, Predict Albedo Distance, Origin Shape Distance, Predict Shape Distance'
        mean = self.all_scores.mean(0)
        std = self.all_scores.std(0)
        header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
        header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
        utils.save_scores(path, self.all_scores, header=header)
