# LatentFace

## Environment

We build the environment with conda on Ubuntu 20.04.1. You can use the following command to build the environment.

```bash
conda create -n latentface python=3.10
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install seaborn==0.13.0 pandas==1.5.3 scikit-learn==1.3.0 pytorch_lightning==1.9.0 diffusers==0.14.0 scikit-image==0.20.0
pip install opencv-python tqdm tensorboard PyYAML
```

## Datasets

Please download the dataset from  
[AffectNet](http://mohammadmahoor.com/affectnet/), [RAF-DB](http://www.whdeng.cn/RAF/model1.html), and [LFW/SLLFW](http://www.whdeng.cn/SLLFW/). Check your dataset if it match the following format and edit the path in the `identity_test.py` and `configs/train_expr_*.yml` files.

```bash

AffectNet
├── Manually_Annotated_file_lists
│   ├── training.csv
│   └── validation.csv
└── Manually_Annotated_Images

RAF-DB
├── aligned
└── list_patition_label.txt

Aff-Wild
├── annotations
│   └── EXPR_Set
│       ├── Train_Set
│       └── Validation_Set
└── cropped_aligned

LFW/SLLFW
├── lfw_crop
├── pair_SLLFW.txt
└── pairs.txt
```

## Pretrained Model and Checkpoints

Please download the pretrained model and the downstream checkpoints from [here](https://www.dropbox.com/scl/fo/apaxoniviytfycfqn9o1n/ACnKre_5VEVvFKE4rlFi3Us?rlkey=58aa1lyabrh0ksjrqsg75fe5f&st=kmzccn31&dl=0).

## Test Performance


<details> <summary> AffectNet </summary>

```bash
python datasets/affectnet_pickle.py --downsample
python expr_test.py --checkpoint checkpoints/affectnet.ckpt --config configs/train_expr_affectnet.yml --gpus 1
```

</details>

<details> <summary> AffectNet Pose 30 Subset </summary>

```bash
python datasets/affectnet_pickle.py --select_path datasets/pose_30_affectnet_list.txt
python expr_test.py --checkpoint checkpoints/affectnet.ckpt --config configs/train_expr_affectnet.yml --gpus 1
```

</details>

<details> <summary> AffectNet Pose 45 Subset</summary>

```bash
python datasets/affectnet_pickle.py --select_path datasets/pose_45_affectnet_list.txt
python expr_test.py --checkpoint checkpoints/affectnet.ckpt --config configs/train_expr_affectnet.yml --gpus 1
```

</details>


<details> <summary> RAF-DB </summary>

```bash
python expr_test.py --checkpoint checkpoints/rafdb.ckpt --config configs/train_expr_rafdb.yml --gpus 1
```

</details>

<details> <summary> Aff-Wild2</summary>

```bash
python expr_test.py --checkpoint checkpoints/affwild.ckpt --config configs/train_expr_affwild.yml --gpus 1
```

</details>

<details> <summary> LFW</summary>

```bash
python identity_test.py --dataset LFW
```

</details>

<details> <summary> SLLFW</summary>

```bash
python identity_test.py --dataset SLLFW
```
</details>

## Finetuning

<details> <summary> AffectNet</summary>

```bash
python datasets/affectnet_pickle.py --downsample
python expr_test.py --train --config configs/train_expr_affectnet.yml --max_epochs 20 --gpus 1
```

</details>

<details> <summary> RAF-DB</summary>

```bash
python expr_test.py --train --config configs/train_expr_rafdb.yml --max_epochs 20 --gpus 1
```

</details>

<details> <summary> Aff-Wild2</summary>

```bash
python expr_test.py --train --config configs/train_expr_affwild.yml --max_epochs 20 --gpus 1
```

</details>