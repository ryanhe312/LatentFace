from PIL import Image
import random
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

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds:(i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(model, predicts, test):
    # eval mlp classifier
    model.eval()
    with torch.no_grad():
        f1 = predicts[0][test]
        f2 = predicts[1][test]
        labels = predicts[2][test]
        inputs = torch.cat([(f1-f2).abs(),(f1+f2).abs(),(f1*f2).abs()],dim=1)
        print(inputs.shape, labels.shape)
        out = model(torch.tensor(inputs).float().to('cuda'))
        out = out.argmax(1).cpu()
        print(out, labels)
        acc = (out == labels).float().mean()
    return acc

import torch.nn as nn
import numpy as np
def find_best_model(predicts,train):
    # train a mlp classifier
    print(predicts[0].shape)
    input_dim = predicts[0].shape[1]*3

    mlp = nn.Sequential(nn.Linear(input_dim, 2))
    mlp.to('cuda')
    mlp.train()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    from tqdm.auto import tqdm
    for epoch in tqdm(range(300)):
        train = np.random.permutation(train)
        for i in range(0, len(predicts), 128):
            f1 = predicts[0][train][i:i + 128]
            f2 = predicts[1][train][i:i + 128]
            labels = predicts[2][train][i:i + 128]
            inputs = torch.cat([(f1-f2).abs(),(f1+f2).abs(),(f1*f2).abs()],dim=1)
            optimizer.zero_grad()
            out = mlp(torch.tensor(inputs).float().to('cuda'))
            loss = loss_fn(out, torch.tensor(labels).to('cuda'))
            loss.backward()
            optimizer.step()
    return mlp


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

def eval(model):
    predicts = [[],[],[]]

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

    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ])

    print('Extracting features from {}...'.format(args.dataset))
    with torch.no_grad():
        batch_size = 128
        for i in tqdm(range(0, num_test, batch_size)):
            img1 = []
            img2 = []
            flag = []

            for j in range(i, min(i + batch_size, num_test)):
                if args.dataset == "SLLFW":
                    sameflag, name1, name2 = get_sllfw_line(pairs_lines, j)
                else:
                    sameflag, name1, name2 = get_lfw_line(pairs_lines, j)
                img1.append(transform(Image.open(os.path.join(root,name1))).unsqueeze(0))
                img2.append(transform(Image.open(os.path.join(root,name2))).unsqueeze(0))
                flag.append(sameflag)

            img1 = torch.cat(img1).to('cuda')
            img2 = torch.cat(img2).to('cuda')

            f1 = model(img1).detach().to('cpu')
            f1_flip = model(torch.flip(img1, [3])).detach().to('cpu')
            f1 = torch.cat([f1, f1_flip], dim=1)
            f2 = model(img2).detach().to('cpu')
            f2_flip = model(torch.flip(img2, [3])).detach().to('cpu')
            f2 = torch.cat([f2, f2_flip], dim=1)
            # print(f1.shape, f2.shape, len(flag))
            predicts[0].extend(f1)
            predicts[1].extend(f2)
            predicts[2].extend(flag)
            
    predicts[0] = torch.stack(predicts[0], dim=0)
    predicts[1] = torch.stack(predicts[1], dim=0)
    predicts[2] = torch.tensor(predicts[2])
    # print(predicts[0].shape, predicts[1].shape, predicts[2].shape)
    # print(predicts)

    accuracy = []
    folds = KFold(n=num_test, n_folds=n_folds)
    for train, test in tqdm(folds):
        print('Evaluate on {}, fold {}'.format(args.dataset, len(accuracy) + 1))
        best_model = find_best_model(predicts, train)
        acc = eval_acc(best_model, predicts, test)
        print('ACC={:.4f}'.format(acc))
        accuracy.append(acc)
    print('Total ACC={:.4f} std={:.4f} max={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.max(accuracy)))

    print(accuracy)

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    backbone=None
    np.random.seed(1234)
    torch.manual_seed(1234)

    from unsup3d import Unsup3D_diffusion
    config = yaml.safe_load(open('configs/train_celeba.yml'))
    model = Unsup3D_diffusion(config)
    state_dict = torch.load('diffusion_64.pth')
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