# LatentFace

We propose a novel unsupervised disentangling framework for facial expression and identity representations. We suggest that the disentangling should be performed in latent space and propose a novel 3D-ware latent diffusion model. Please refer to [our paper](https://arxiv.org/abs/2309.08273) for more details.

## Environment

You can further use the following commands to install extra packages for training, or follow [the official tutorial](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install Pytorch3D.

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install lpips
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

## Pretrained Models and Datasets

Please download the CelebA-pretrained [Unsup3D](https://github.com/elliottwu/unsup3d) model from the [link](https://www.robots.ox.ac.uk/~vgg/research/unsup3d/data/pretrained_celeba.zip) and put `checkpoint030.pth` in the `results/diffusion` folder. Then run the following command to convert the checkpoint.

```bash
python convert.py results/diffusion/checkpoint030.pth
```

Please download the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/) dataset from the [link](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/zippedFaces.tar.gz) and replace the path in `configs.yml`.

## Training

You can train the model with the following command.

```bash 
python run_diffusion.py --config configs.yml
```

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