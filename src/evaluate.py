import datasets
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformers import AutoTokenizer

from models.kd_model import KDModel
from utils import get_tensordataset

# Load dataset
dataset = datasets.load_dataset('glue', 'cola')
train_feats = dataset['train']['sentence']
train_labels = dataset['train']['label']
validation_feats = dataset['validation']['sentence']
validation_labels = dataset['validation']['label']
test_feats = dataset['test']['sentence']
test_labels = dataset['test']['label']

train_data = Dataset.from_dict({
    'text': train_feats,
    'label': train_labels
})
validation_data = Dataset.from_dict({
    'text': validation_feats,
    'label': validation_labels
})
test_data = Dataset.from_dict({
    'text': test_feats,
    'label': test_labels
})

# Tokenizers
student_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
teacher_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Dataloaders
train_dataset = get_tensordataset(
    data=train_data,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=32
)
validation_dataset = get_tensordataset(
    data=validation_data,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=32
)
test_dataset = get_tensordataset(
    data=test_data,
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=32
)

# Setup and load checkpoint
model = KDModel.load_from_checkpoint(
    checkpoint_path='../cola_train_loss=0.10-val_loss=0.05.ckpt',
    initial_embedding_model='distilbert-base-uncased',
    teacher_model='bert-base-uncased',
    num_feats=768,
    edge_construction_hidden_dims=[],
    feature_construction_hidden_dims=[],
    gcn_hidden_dims=[],
    feature_construction_output_dim=512,
    gcn_output_dim=768,
    cache_dir='/root/.cache/huggingface/transformers',
    batch_size=32,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    num_workers=16
)
model.eval()

# Get traininig dataset embeddings
train_initial_embeddings, train_student_embeddings, train_teacher_embeddings = [], [], []
for batch in tqdm(train_dataloader, desc='Training embeddings'):
    with torch.no_grad():
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch
        train_initial_embedding = model.initial_embedding_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask
        ).last_hidden_state
        train_student_embedding = model.student_model(
            train_initial_embedding,
            student_attention_mask
        ).mean(1)
        train_student_embeddings.extend(list(train_student_embedding.numpy()))
        train_initial_embedding = train_initial_embedding[:, 0, :]
        train_initial_embeddings.extend(list(train_initial_embedding.numpy()))
        train_teacher_embedding = model.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask
        ).last_hidden_state[:, 0, :]
        train_teacher_embeddings.extend(list(train_teacher_embedding.numpy()))

# Get validation dataset embeddings
validation_initial_embeddings, validation_student_embeddings, validation_teacher_embeddings = [], [], []
for batch in tqdm(validation_dataloader, desc='Validation embeddings'):
    with torch.no_grad():
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch
        validation_initial_embedding = model.initial_embedding_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask
        ).last_hidden_state
        validation_student_embedding = model.student_model(
            validation_initial_embedding,
            student_attention_mask
        ).mean(1)
        validation_student_embeddings.extend(
            list(validation_student_embedding.numpy()))
        validation_initial_embedding = validation_initial_embedding[:, 0, :]
        validation_initial_embeddings.extend(
            list(validation_initial_embedding.numpy()))
        validation_teacher_embedding = model.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask
        ).last_hidden_state[:, 0, :]
        validation_teacher_embeddings.extend(
            list(validation_teacher_embedding.numpy()))

# Get test dataset embeddings
test_initial_embeddings, test_student_embeddings, test_teacher_embeddings = [], [], []
for batch in tqdm(test_dataloader, desc='Test embeddings'):
    with torch.no_grad():
        student_input_ids, student_attention_mask, teacher_input_ids, teacher_attention_mask = batch
        test_initial_embedding = model.initial_embedding_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask
        ).last_hidden_state
        test_student_embedding = model.student_model(
            test_initial_embedding,
            student_attention_mask
        ).mean(1)
        test_student_embeddings.extend(list(test_student_embedding.numpy()))
        test_initial_embedding = test_initial_embedding[:, 0, :]
        test_initial_embeddings.extend(list(test_initial_embedding.numpy()))
        test_teacher_embedding = model.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask
        ).last_hidden_state[:, 0, :]
        test_teacher_embeddings.extend(list(test_teacher_embedding.numpy()))

# Train and evaluate decision tree classifier for initial embedding model
clf = DecisionTreeClassifier()
clf.fit(train_initial_embeddings, train_labels)
validation_preds = clf.predict(validation_initial_embeddings)
initial_embedding_model_f1 = f1_score(
    validation_labels, validation_preds, average='macro')
initial_embedding_model_accuracy = accuracy_score(
    validation_labels, validation_preds)

# Train and evaluate decision tree classifier for student embedding model
clf = DecisionTreeClassifier()
clf.fit(train_student_embeddings, train_labels)
validation_preds = clf.predict(validation_student_embeddings)
student_embedding_model_f1 = f1_score(
    validation_labels, validation_preds, average='macro')
student_embedding_model_accuracy = accuracy_score(
    validation_labels, validation_preds)

# Train and evaluate decision tree classifier for teacher embedding model
clf = DecisionTreeClassifier()
clf.fit(train_teacher_embeddings, train_labels)
validation_preds = clf.predict(validation_teacher_embeddings)
teacher_embedding_model_f1 = f1_score(
    validation_labels, validation_preds, average='macro')
teacher_embedding_model_accuracy = accuracy_score(
    validation_labels, validation_preds)

print(
    f'Initial embedding model\tF1 Macro: {initial_embedding_model_f1:.4f}\tAccuracy: {initial_embedding_model_accuracy:.4f}')
print(
    f'Student embedding model\tF1 Macro: {student_embedding_model_f1:.4f}\tAccuracy: {student_embedding_model_accuracy:.4f}')
print(
    f'Teacher embedding model\tF1 Macro: {teacher_embedding_model_f1:.4f}\tAccuracy: {teacher_embedding_model_accuracy:.4f}')
