from argparse import ArgumentParser
import pytorch_lightning as pl
import yaml

from datasets import *
from model import *

def cli_main():
    pl.seed_everything(1234, workers=True)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model_args = yaml.safe_load(open(args.config, 'r')) if args.config else {}
    if "RAF-DB" in model_args.get('dataset_dir', ''):
        dm = RAFDBDataModule(model_args)
    elif "AffectNet" in model_args.get('dataset_dir', ''):
        dm = AffectNetDataModule(model_args)
    elif "Aff-Wild" in model_args.get('dataset_dir', ''):
        dm = AffWild2DataModule(model_args)
    else:
        raise ValueError("dataset not supported")

    model = ExprClassifier.load_from_checkpoint(args.checkpoint) \
        if args.checkpoint else ExprClassifier(model_args)

    # ------------
    # training
    # ------------
    if args.train: trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()