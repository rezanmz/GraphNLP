import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import DistilBertModel, DistilBertTokenizer, RobertaTokenizer, RobertaModel
from typing import List
from models.student_model import StudentModel


class KDModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        num_feats: int,
        edge_construction_hidden_dims: List[int],
        feature_construction_hidden_dims: List[int],
        gcn_hidden_dims: List[int],
        feature_construction_output_dim: int,
        gcn_output_dim: int,
        train_data: List[str],
        val_data: List[str]
    ):
        super().__init__()

        self.student_tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self.initial_embedding_model = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        self.student_model = StudentModel(
            num_feats,
            edge_construction_hidden_dims,
            feature_construction_hidden_dims,
            gcn_hidden_dims,
            feature_construction_output_dim,
            gcn_output_dim
        )

        self.teacher_tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-base')
        self.teacher_model = RobertaModel.from_pretrained('roberta-base')

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch
        # student_tokenized_sentence = self.student_tokenizer(
        #     sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
        # teacher_tokenized_sentence = self.teacher_tokenizer(
        #     sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Teacher model
        with torch.no_grad():
            teacher_output = self.teacher_model(
                input_ids=teacher_input_ids, attention_mask=teacher_attention_mask).last_hidden_state[:, 0, :]

        # Student model
        with torch.no_grad():
            initial_embeddings = self.initial_embedding_model(
                input_ids=student_input_ids, attention_mask=student_attention_mask).last_hidden_state
        student_output = self.student_model(initial_embeddings).mean(1)

        loss = F.mse_loss(student_output, teacher_output)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch
        # sentence = batch
        # student_tokenized_sentence = self.student_tokenizer(
        #     sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
        # teacher_tokenized_sentence = self.teacher_tokenizer(
        #     sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Teacher model
        with torch.no_grad():
            teacher_output = self.teacher_model(
                input_ids=teacher_input_ids, attention_mask=teacher_attention_mask).last_hidden_state[:, 0, :]

        # Student model
        with torch.no_grad():
            initial_embeddings = self.initial_embedding_model(
                input_ids=student_input_ids, attention_mask=student_attention_mask).last_hidden_state
        student_output = self.student_model(initial_embeddings).mean(1)

        loss = F.mse_loss(student_output, teacher_output)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def train_dataloader(self):
        student_data = self.train_data.map(lambda e: self.student_tokenizer(
            e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        student_data.set_format(type='torch', columns=[
                                'input_ids', 'attention_mask', 'label'])
        teacher_data = self.train_data.map(lambda e: self.teacher_tokenizer(
            e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        teacher_data.set_format(type='torch', columns=[
                                'input_ids', 'attention_mask', 'label'])
        dataset = torch.utils.data.TensorDataset(
            student_data['input_ids'], student_data['attention_mask'], teacher_data['input_ids'], teacher_data['attention_mask'])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=48)

    def val_dataloader(self):
        student_data = self.val_data.map(lambda e: self.student_tokenizer(
            e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        student_data.set_format(type='torch', columns=[
                                'input_ids', 'attention_mask', 'label'])
        teacher_data = self.val_data.map(lambda e: self.teacher_tokenizer(
            e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        teacher_data.set_format(type='torch', columns=[
                                'input_ids', 'attention_mask', 'label'])
        dataset = torch.utils.data.TensorDataset(
            student_data['input_ids'], student_data['attention_mask'], teacher_data['input_ids'], teacher_data['attention_mask'])
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=48)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.student_model.parameters())
