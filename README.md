# LatentFace

We propose a novel unsupervised disentangling framework for facial expression and identity representations. We suggest that the disentangling should be performed in latent space and propose a novel 3D-ware latent diffusion model. Please refer to [our paper](https://arxiv.org/abs/2309.08273) for more details.

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
[AffectNet](http://mohammadmahoor.com/affectnet/), [RAF-DB](http://www.whdeng.cn/RAF/model1.html), [Aff-wild2](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/), and [LFW/SLLFW](http://www.whdeng.cn/SLLFW/). Check your dataset if it match the following format and edit the path in the `identity_test.py` and `configs/train_expr_*.yml` files.

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

Please download the pretrained LatentFace model from [here](https://www.dropbox.com/scl/fi/5duiidexvrw00fx2ex2gp/latentface.7z?rlkey=n9ruy237at44z8ejhlnv398hx&dl=0) and put it in the `model/latentface` folder.

Please download the checkpoints model from [here](https://www.dropbox.com/scl/fo/zpn3h1yv20fg0yhgku7hp/h?rlkey=b676nkjr1ghl1jyb1gp1l7jb0&dl=0) and put it in the `checkpoints` folder.


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

## Pretraining

Please refer to [this branch](https://github.com/ryanhe312/LatentFace/tree/pretrain) for pretraining codes.

## Citation

If you find this work useful, please cite our paper with the following bibtex:

```bibtex
@misc{he2023unsupervised,
      title={Unsupervised Disentangling of Facial Representations with 3D-aware Latent Diffusion Models}, 
      author={Ruian He and Zhen Xing and Weimin Tan and Bo Yan},
      year={2023},
      eprint={2309.08273},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```