import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchmetrics as tm
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import yaml
from torch.nn import functional as F
from .metrics import *

affectnet_list = ["neutral", "happiness", "sadness", "surprise" ,  "fear",  "disgust", "anger", "contempt"]
raf_db_list = ["neutral", "surprised", "fearful", "disgusted", "happy", "sad", "angry"]
affwild_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']

class ExprClassifier(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        from .latentface.model_diffusion import Unsup3D_diffusion
        config = yaml.safe_load(open('model/latentface/train_celeba.yml'))
        self.model = Unsup3D_diffusion(config)
        state_dict = torch.load('model/latentface/diffusion_64_depth.pth')
        self.model.load_model_state(state_dict)
        self.model.to_device('cuda')
        self.model.set_eval()
        self.model.netEA.requires_grad_(False)
        self.model.netED.requires_grad_(False)
        self.model.netA.requires_grad_(False)
        self.model.netD.requires_grad_(False)
        input_dim = 256 * 4

        class_num = 8 if "AffectNet" in self.hparams.dataset_dir else 7

        if self.hparams.data_type == 'VA_Set':
            self.head = nn.Sequential(nn.BatchNorm1d(input_dim),torch.nn.Linear(input_dim, 2,bias=False))
            self.loss = nn.SmoothL1Loss()
        elif self.hparams.data_type == 'EXPR_Set':
            self.head = nn.Sequential(nn.BatchNorm1d(input_dim),torch.nn.Linear(input_dim, class_num, bias=False))
            self.loss = nn.CrossEntropyLoss()
        else:
            self.head = nn.Sequential(nn.BatchNorm1d(input_dim),torch.nn.Linear(input_dim, 12,bias=False))
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        if hasattr(self, 'model') and hasattr(self.model, 'scheduler'):
            albedo = self.model.netA(x*2-1)[0]
            shape = self.model.netD(x*2-1)[0]

            neutral_a = torch.randn(albedo.shape).to(albedo.device)
            for t in self.model.scheduler.timesteps:
                concat_input = torch.cat([neutral_a, albedo], dim=1)
                model_output = self.model.netEA(concat_input, t).sample
                neutral_a = self.model.scheduler.step(model_output, t, neutral_a, eta=0).prev_sample
                                          
            neutral_d = torch.randn(shape.shape).to(shape.device)
            for t in self.model.scheduler.timesteps:
                concat_input = torch.cat([neutral_d, shape], dim=1)
                model_output = self.model.netED(concat_input, t).sample
                neutral_d = self.model.scheduler.step(model_output, t, neutral_d, eta=0).prev_sample
            embedding = torch.cat([(albedo - neutral_a).squeeze(),albedo.squeeze(),(shape - neutral_d).squeeze(),shape.squeeze()],-1)
        else:
            embedding = self.backbone(x)
        # print(embedding.shape)
        y_hat = self.head(embedding)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.to(torch.long))
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.to(torch.long))

        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        self.log_dict({'val/loss': loss})

        return y_hat, y

    def validation_epoch_end(self, outputs) -> None:
        y_hat = []
        y = []

        for step in outputs:
            y_hat.append(step[0])
            y.append(step[1])

        y_hat = np.concatenate(y_hat, axis=0)
        y = np.concatenate(y, axis=0)

        if self.hparams.data_type == 'VA_Set':
            item, sum = VA_metric(y, y_hat)
            self.log_dict({'val/CCC-V': item[0], 'val/CCC-A': item[1], 'val/score': sum})

        elif self.hparams.data_type == 'EXPR_Set':
            f1_acc, score, matrix = EXPR_metric(y, y_hat)
            self.log_dict({'val/f1': f1_acc[0], 'val/acc': f1_acc[1], 'val/score': score})
            print(f1_acc)

            if "RAF-DB" in self.hparams.dataset_dir:
                labels_ticks = raf_db_list
            elif "AffectNet" in self.hparams.dataset_dir:
                labels_ticks = affectnet_list
            elif "Aff-Wild" in self.hparams.dataset_dir:
                labels_ticks = affwild_list
            else:
                raise ValueError("dataset not supported")

            fig, ax = plt.subplots()
            ax = sns.heatmap(matrix, cmap='Blues', annot=True, fmt='.2f',
                        xticklabels=labels_ticks, yticklabels=labels_ticks)
            self.logger.experiment.add_figure('val/conf', fig, self.current_epoch)

        else:
            f1_acc, score = AU_metric(y, y_hat)
            self.log_dict({'val/f1': f1_acc[0], 'val/acc': f1_acc[1], 'val/score': score})

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y_hat = self(batch)
        y_hat = y_hat.detach().cpu().numpy()

        if self.hparams.data_type == 'EXPR_Set':
            y_hat = np.argmax(y_hat, axis=-1)

        elif self.hparams.data_type == 'AU_Set':
            y_hat = (y_hat > 0.5).astype(int)

        else:
            y_hat = np.clip(y_hat, -1 ,1)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.hparams.get('learning_rate', 1e-3))
        return [optimizer]

    def configure_callbacks(self):
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val/acc',
            mode='max',
            filename='epoch{epoch}-loss{val/loss:.2f}-score{val/score:.2f}',
            save_top_k=3,
            auto_insert_metric_name=False
        )
        return [checkpoint]
