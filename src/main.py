import argparse
from ast import arg
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, logging

from models.kd_model import KDModel
from utils import get_dataloader, load_datasets

logging.set_verbosity_error()


def run(**kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = load_datasets(
        kwargs['datasets'],
        cache_dir=kwargs['cache_dir']
    )
    train_data = dataset['train']
    validation_data = dataset['validation']

    # Tokenizers
    student_tokenizer = AutoTokenizer.from_pretrained(
        kwargs['initial_embedding_model'])
    teacher_tokenizer = AutoTokenizer.from_pretrained(kwargs['teacher_model'])

    # Dataloaders
    train_dataloader = get_dataloader(
        data=train_data,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=kwargs['num_workers']
    )
    validation_dataloader = get_dataloader(
        data=validation_data,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        batch_size=kwargs['batch_size'],
        shuffle=False,
        num_workers=kwargs['num_workers']
    )

    model = KDModel(
        initial_embedding_model=kwargs['initial_embedding_model'],
        teacher_model=kwargs['teacher_model'],
        num_feats=kwargs['num_feats'],  # must match initial_embedding_model
        edge_construction_hidden_dims=kwargs['edge_construction_hidden_dims'],
        feature_construction_hidden_dims=kwargs['feature_construction_hidden_dims'],
        gcn_hidden_dims=kwargs['gcn_hidden_dims'],
        feature_construction_output_dim=kwargs['feature_construction_output_dim'],
        gcn_output_dim=kwargs['gcn_output_dim'],  # must match teacher_model
    )

    logger = WandbLogger(
        project=kwargs['wandb_project'],
        offline=kwargs['offline']
    )
    logger.watch(model)

    trainer = pl.Trainer(
        max_epochs=kwargs['max_epochs'],
        accelerator="gpu",
        devices=kwargs['num_gpus'],
        num_nodes=kwargs['num_nodes'],
        logger=logger
    )

    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--initial-embedding-model',
                           type=str, default='distilbert-base-uncased')
    argparser.add_argument('--teacher-model', type=str,
                           default='bert-base-uncased')
    argparser.add_argument('--batch-size', type=int, default=4)
    argparser.add_argument('--num-workers', type=int, default=4)
    argparser.add_argument('--datasets', nargs='+', default=['ag_news'])
    argparser.add_argument('--num-feats', type=int, default=768)
    argparser.add_argument(
        '--edge-construction-hidden-dims', nargs='+', default=[])
    argparser.add_argument(
        '--feature-construction-hidden-dims', nargs='+', default=[])
    argparser.add_argument('--gcn-hidden-dims', nargs='+', default=[])
    argparser.add_argument(
        '--feature-construction-output-dim', type=int, default=512)
    argparser.add_argument('--gcn-output-dim', type=int, default=768)
    argparser.add_argument('--max-epochs', type=int, default=100)
    argparser.add_argument('--num-gpus', type=int, default=1)
    argparser.add_argument('--num-nodes', type=int, default=1)
    argparse.add_argument('--huggingface-cache-dir',
                          type=str, default='~/.cache/huggingface/datasets')
    argparser.add_argument('--wandb-project', type=str, default='graph-nlp')
    argparser.add_argument('--offline', action='store_true')
    argparser.add_argument('--no-offline', action='store_false')
    argparser.set_defaults(offline=False)

    args = argparser.parse_args()
    run(**vars(args))
