import pickle
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

isUnix = True
affectnet_dir = '/home/user2/dataset/face/AffectNet/'

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action='store_true', help='whether to visualize the distribution')
parser.add_argument('--downsample', action='store_true', help='whether to downsample training set')
parser.add_argument('--upsample', action='store_true', help='whether to upsample training set')
parser.add_argument('--csv_file_dir', type=str, default=os.path.join(affectnet_dir,'Manually_Annotated_file_lists'))
parser.add_argument('--img_dir', type=str, default=os.path.join(affectnet_dir,'Manually_Annotated_Images'))
parser.add_argument('--distribution_output', type=str, default=os.path.join(affectnet_dir,'AffectNet.distribution.jpg'))
parser.add_argument('--save_path', type=str, default=os.path.join(affectnet_dir,'annotation_split.pkl'))
parser.add_argument('--select_path', type=str, default=None)
args = parser.parse_args()
expr_list = ["neutral", "happiness", "sadness", "surprise" ,  "fear",  "disgust", "anger", "contempt"]

TYPE = {
    'training.csv' : 'Train_Set',
    'validation.csv' : 'Validation_Set'
}

def read_csv():
    files = os.listdir(args.csv_file_dir)
    dire = []
    expr = []
    dict = {'Train_Set':{}, 'Validation_Set':{}}
    for file in tqdm(files):
        obj = pd.read_csv(os.path.join(args.csv_file_dir, file))
        # filter data
        obj = obj[
            obj["expression"].between(0, 7) & obj["valence"].between(-1.0, 1.0) & obj["arousal"].between(-1.0, 1.0)]
        dire = obj["subDirectory_filePath"].values
        expr = obj["expression"].values

        x1 = obj["face_x"].values
        y1 = obj["face_y"].values
        x2 = obj["face_width"].values + x1
        y2 = obj["face_height"].values + y1

        # update path
        ## Linux
        if isUnix:
            prefix = '/'.join(args.img_dir.split("/"))
            path = [prefix + '/' + subdire for subdire in dire]
        ## Windows
        else:
            prefix = '\\'.join(args.img_dir.split("\\")) 
            path = [prefix + '\\' + subdire.replace('/','\\') for subdire in dire]

        # update dict
        dict[TYPE[file]]["path"] = path
        dict[TYPE[file]]["label"] = np.array(expr)
        dict[TYPE[file]]["face"] = np.stack([
            np.array(x1),np.array(y1),
            np.array(x2),np.array(y2)],axis=-1)
        # print(dict[TYPE[file]]["face"].shape)
        # dtype=int64
        # dict[TYPE[file]]["valence"] = np.array(valence)
        # dtype=float64
        # dict[TYPE[file]]["arousal"] = np.array(arousal)
        # dtype=float64

        if args.select_path and TYPE[file] == 'Validation_Set':
            with open(args.select_path) as f:
                ## Linux
                if isUnix:
                    select = [line.strip().split('/')[1].split('.')[0]+'.jpg' for line in f.readlines()]
                ## Windows
                else:
                    select = [line.strip().split('/')[1].split('.')[0].replace('/', '\\')+'.jpg' for line in f.readlines()]
            # print(select)
            # print(dict[TYPE[file]]["label"].shape)
            for i in reversed(range(len(dict[TYPE[file]]["path"]))):
                ## Linux
                if isUnix:
                    if dict[TYPE[file]]["path"][i].split('/')[-1] not in select:
                        dict[TYPE[file]]["path"].pop(i)
                        dict[TYPE[file]]["label"] = np.delete(dict[TYPE[file]]["label"], i, 0)
                        dict[TYPE[file]]["face"] = np.delete(dict[TYPE[file]]["face"], i, 0)
                ## Windows
                else:
                    if dict[TYPE[file]]["path"][i].split('\\')[-1] not in select:
                        # print(dict[TYPE[file]]["path"][i].split('/')[-1])
                        dict[TYPE[file]]["path"].pop(i)
                        dict[TYPE[file]]["label"] = np.delete(dict[TYPE[file]]["label"], i, 0)
                        dict[TYPE[file]]["face"] = np.delete(dict[TYPE[file]]["face"], i, 0)
                        # np.delete(dict[TYPE[file]]["valence"], 0, i)
                        # np.delete(dict[TYPE[file]]["arousal"], 0, i)

    # ====================downsample===================================

    def downsample(data_merged, idx, times):
        labels = data_merged['label']
        is_drop = labels == idx
        keep = np.array([True if index % times == 0 else False for index in range(len(labels))])
        to_drop = is_drop & ~keep
        labels = data_merged['label'][~to_drop]
        face = data_merged['face'][~to_drop]
        paths = np.array(data_merged['path'])[~to_drop]
        data_merged.update({'label': labels, 'path': paths, 'face':face})

    if args.downsample:
        data_merged = dict['Train_Set']
        downsample(data_merged, 0, 10) # neutral 1/10
        downsample(data_merged, 1, 20) # happy 1/20
        downsample(data_merged, 2, 4) # sad 1/4
        downsample(data_merged, 3, 2) # surprise 1/2
        downsample(data_merged, 6, 4) # anger 1/4

    # ====================upsample===================================
    def upsample(data_merged, idx, times):
        labels = data_merged['label']
        is_add = labels == idx
        labels = data_merged['label'].copy()
        face = data_merged['face'].copy()
        paths = np.array(data_merged['path']).copy()
        for i in range(times):
            labels = np.concatenate([labels, data_merged['label'][is_add]])
            paths = np.concatenate([paths, np.array(data_merged['path'])[is_add]])
            face = np.concatenate([face, np.array(data_merged['face'])[is_add]])
        data_merged.update({'label': labels, 'path': paths, 'face':face})

    if args.upsample:
        data_merged = dict['Train_Set']
        downsample(data_merged, 1, 2) # happy 1/2
        upsample(data_merged, 2, 3) # sad 3
        upsample(data_merged, 3, 5) # surprise 5
        upsample(data_merged, 4, 10) # fear 10
        upsample(data_merged, 5, 20) # disgust 20
        upsample(data_merged, 6, 3)  # anger 3
        upsample(data_merged, 7, 20)  # contempt 20

    # save file
    df = pd.DataFrame.from_dict(dict)
    pickle.dump(df, open(args.save_path, 'wb'))
    return df


def plot_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['Train_Set']['label']
    for i in range(8):
        find_true = sum(all_samples == i)
        histogram[i] = find_true / all_samples.shape[0]
    plt.bar(np.arange(len(expr_list)), histogram)
    plt.xticks(np.arange(len(expr_list)), expr_list)
    plt.savefig(args.distribution_output)
    plt.show()


def print_distribution(data_file):
    histogram = np.zeros(len(expr_list))
    all_samples = data_file['Train_Set']['label']
    for i in range(8):
        find_true = sum(all_samples == i)
        histogram[i] = find_true
    print(expr_list)
    print(histogram)


if __name__ == "__main__":
    data_file = read_csv()
    plot_distribution(data_file)
    print_distribution(data_file)
    print(len(data_file['Train_Set']['path']),len(data_file['Validation_Set']['path']))
    print(len(data_file['Train_Set']['label']), len(data_file['Validation_Set']['label']))
