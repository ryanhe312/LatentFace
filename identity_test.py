from PIL import Image
import numpy as np
import yaml
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description='test lfw and sllfw')
parser.add_argument('--dataset', type=str, default="LFW", help='LFW or SLLFW')
parser.add_argument('--path', type=str, default="/home/user2/dataset/face/LFW/", help='dataset path')
args = parser.parse_args()

cudnn.benchmark = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

img_size = 64

def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])

    img, img_ = transform(img), transform(F.hflip(img))

    # Model
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds:(i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def get_lfw_line(pairs_lines, i):
    p = pairs_lines[i].replace('\n', '').split('\t')

    if 3 == len(p):
        sameflag = 1
        name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
    elif 4 == len(p):
        sameflag = 0
        name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
    else:
        raise ValueError("WRONG LINE IN 'pairs.txt! ")

    return sameflag,name1,name2

def get_cplfw_line(pairs_lines, i):
    p1 = pairs_lines[2 * i].replace('\n', '').split(' ')
    p2 = pairs_lines[2 * i + 1].replace('\n', '').split(' ')

    assert p1[1]==p2[1], "The labels of line1 and line2 don't match"
    sameflag = int(p1[1]!='0')
    name1 = p1[0]#.split('.')[0]+'_recrop.png'
    name2 = p2[0]#.split('.')[0]+'_recrop.png'
    return sameflag,name1,name2

def get_sllfw_line(pairs_lines, i):
    p1 = pairs_lines[2 * i].replace('\n', '').split('/')
    p2 = pairs_lines[2 * i + 1].replace('\n', '').split('/')

    sameflag = int(p1[0]==p2[0])
    name1 = p1[0]+'/'+p1[1]
    name2 = p2[0]+'/'+p2[1]
    return sameflag,name1,name2

from tqdm.auto import tqdm

def eval(model, is_gray=False):
    predicts = []

    # LFW
    root = os.path.join(args.path,'lfw_crop')
    if args.dataset == "SLLFW":
        pairs_path = os.path.join(args.path,'pair_SLLFW.txt')  
        with open(pairs_path) as f:
            pairs_lines = f.readlines()
    else:
        pairs_path = os.path.join(args.path,'pairs.txt')
        with open(pairs_path) as f:
            pairs_lines = f.readlines()[1:]

    num_test = 6000
    n_folds = 10

    with torch.no_grad():
        for i in tqdm(range(num_test)):
            if args.dataset == "SLLFW":
                sameflag, name1, name2 = get_sllfw_line(pairs_lines, i)
            else:
                sameflag, name1, name2 = get_lfw_line(pairs_lines, i)

            img1 =  Image.open(os.path.join(root,name1))
            img2 =  Image.open(os.path.join(root,name2))

            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=num_test, n_folds=n_folds)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))

    for train, test in tqdm(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f} max={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd), np.max(accuracy)))

    print(accuracy)

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    backbone=None
    np.random.seed(1234)
    torch.manual_seed(1234)

    from model.latentface.model_diffusion import Unsup3D_diffusion
    config = yaml.safe_load(open('model/latentface/train_celeba.yml'))
    model = Unsup3D_diffusion(config)
    state_dict = torch.load('model/latentface/diffusion_64_depth.pth')
    model.load_model_state(state_dict)
    model.to_device('cuda')
    model.set_eval()
    model.netA.requires_grad_(False)
    model.netD.requires_grad_(False)
    model.netEA.requires_grad_(False)
    model.netED.requires_grad_(False)

    def backbone(x):
        albedo = model.netA(x*2-1)[0]
        neutral_a = torch.randn(albedo.shape).to(albedo.device)
        for t in model.scheduler.timesteps:
            concat_input = torch.cat([neutral_a, albedo], dim=1)
            model_output = model.netEA(concat_input, t).sample
            neutral_a = model.scheduler.step(model_output, t, neutral_a, eta=0).prev_sample

        shape = model.netD(x*2-1)[0]
        neutral_d = torch.randn(shape.shape).to(shape.device)
        for t in model.scheduler.timesteps:
            concat_input = torch.cat([neutral_d, shape], dim=1)
            model_output = model.netED(concat_input, t).sample
            neutral_d = model.scheduler.step(model_output, t, neutral_d, eta=0).prev_sample
        embedding = torch.cat([neutral_a.squeeze(-1).squeeze(-1), albedo.squeeze(-1).squeeze(-1),neutral_d.squeeze(-1).squeeze(-1), shape.squeeze(-1).squeeze(-1)],-1)
        return embedding

    _, result = eval(backbone)