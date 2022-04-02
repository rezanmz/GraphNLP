import datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.kd_model import KDModel

dataset = datasets.load_dataset('ag_news')
train_data = dataset['train']
val_data = dataset['test']

logger = TensorBoardLogger('logs', name='kd_model')
trainer = pl.Trainer(max_epochs=100, logger=logger,
                     accelerator="gpu", devices=-1, strategy='ddp')
model = KDModel(
    batch_size=32,
    num_feats=768,  # must match initial_embedding_model
    edge_construction_hidden_dims=[],
    feature_construction_hidden_dims=[],
    gcn_hidden_dims=[],
    feature_construction_output_dim=512,
    gcn_output_dim=768,  # must match teacher_model
    train_data=train_data,
    val_data=val_data
)

trainer.fit(model)
