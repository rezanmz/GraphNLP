from typing import List

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from transformers import AutoModel

from models.student_model import StudentModel


class KDModel(pl.LightningModule):
    def __init__(
        self,
        num_feats: int,
        edge_construction_hidden_dims: List[int],
        feature_construction_hidden_dims: List[int],
        gcn_hidden_dims: List[int],
        feature_construction_output_dim: int,
        gcn_output_dim: int,
        initial_embedding_model: str,
        teacher_model: str,
        cache_dir: str,
        batch_size: int,
        train_dataset: torch.utils.data.TensorDataset,
        validation_dataset: torch.utils.data.TensorDataset,
        num_workers: int,
    ):
        super().__init__()

        self.initial_embedding_model = AutoModel.from_pretrained(
            initial_embedding_model,
            cache_dir=cache_dir
        )
        self.student_model = StudentModel(
            num_feats,
            edge_construction_hidden_dims,
            feature_construction_hidden_dims,
            gcn_hidden_dims,
            feature_construction_output_dim,
            gcn_output_dim
        )
        self.teacher_model = AutoModel.from_pretrained(
            teacher_model,
            cache_dir=cache_dir
        )

        self.loss_fn = lambda x, y: 1 - F.cosine_similarity(x, y).mean()

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = validation_dataset
        self.num_workers = num_workers

    def common_step(self, batch, batch_idx):
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch

        # Teacher model
        with torch.no_grad():
            teacher_output = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            ).last_hidden_state[:, 0, :]

        # Student model
        with torch.no_grad():
            initial_embeddings = self.initial_embedding_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            ).last_hidden_state
        student_output = self.student_model(
            initial_embeddings,
            student_attention_mask
        ).mean(1)

        loss = self.loss_fn(student_output, teacher_output)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.student_model.parameters())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
