from typing import List

import datasets
import torch
from datasets import Dataset, DatasetDict


def load_datasets(dataset_names: List[str]):
    train, validation, test = [], [], []
    if 'ag_news' in dataset_names:
        dataset = datasets.load_dataset('ag_news')
        train.extend(dataset['train']['text'])
        test.extend(dataset['test']['text'])
    if 'glue' in dataset_names:
        # cola
        dataset = datasets.load_dataset('glue', 'cola')
        train.extend(dataset['train']['sentence'])
        validation.extend(dataset['validation']['sentence'])
        test.extend(dataset['test']['sentence'])
        # sst2
        dataset = datasets.load_dataset('glue', 'sst2')
        train.extend(dataset['train']['sentence'])
        validation.extend(dataset['validation']['sentence'])
        test.extend(dataset['test']['sentence'])
        # mrpc
        dataset = datasets.load_dataset('glue', 'mrpc')
        train.extend(dataset['train']['sentence1'])
        train.extend(dataset['train']['sentence2'])
        validation.extend(dataset['validation']['sentence1'])
        validation.extend(dataset['validation']['sentence2'])
        test.extend(dataset['test']['sentence1'])
        test.extend(dataset['test']['sentence2'])
        # qqp
        dataset = datasets.load_dataset('glue', 'qqp')
        train.extend(dataset['train']['question1'])
        train.extend(dataset['train']['question2'])
        validation.extend(dataset['validation']['question1'])
        validation.extend(dataset['validation']['question2'])
        test.extend(dataset['test']['question1'])
        test.extend(dataset['test']['question2'])
        # stsb
        dataset = datasets.load_dataset('glue', 'stsb')
        train.extend(dataset['train']['sentence1'])
        train.extend(dataset['train']['sentence2'])
        validation.extend(dataset['validation']['sentence1'])
        validation.extend(dataset['validation']['sentence2'])
        test.extend(dataset['test']['sentence1'])
        test.extend(dataset['test']['sentence2'])
        # mnli
        dataset = datasets.load_dataset('glue', 'mnli')
        train.extend(dataset['train']['premise'])
        train.extend(dataset['train']['hypothesis'])
        validation.extend(dataset['validation_matched']['premise'])
        validation.extend(dataset['validation_matched']['hypothesis'])
        validation.extend(dataset['validation_mismatched']['premise'])
        validation.extend(dataset['validation_mismatched']['hypothesis'])
        test.extend(dataset['test_matched']['premise'])
        test.extend(dataset['test_matched']['hypothesis'])
        test.extend(dataset['test_mismatched']['premise'])
        test.extend(dataset['test_mismatched']['hypothesis'])
        # qnli
        dataset = datasets.load_dataset('glue', 'qnli')
        train.extend(dataset['train']['question'])
        train.extend(dataset['train']['sentence'])
        validation.extend(dataset['validation']['question'])
        validation.extend(dataset['validation']['sentence'])
        test.extend(dataset['test']['question'])
        test.extend(dataset['test']['sentence'])
        # rte
        dataset = datasets.load_dataset('glue', 'rte')
        train.extend(dataset['train']['sentence1'])
        train.extend(dataset['train']['sentence2'])
        validation.extend(dataset['validation']['sentence1'])
        validation.extend(dataset['validation']['sentence2'])
        test.extend(dataset['test']['sentence1'])
        test.extend(dataset['test']['sentence2'])
        # wnli
        dataset = datasets.load_dataset('glue', 'wnli')
        train.extend(dataset['train']['sentence1'])
        train.extend(dataset['train']['sentence2'])
        validation.extend(dataset['validation']['sentence1'])
        validation.extend(dataset['validation']['sentence2'])
        test.extend(dataset['test']['sentence1'])
        test.extend(dataset['test']['sentence2'])
        # ax
        dataset = datasets.load_dataset('glue', 'ax')
        test.extend(dataset['test']['premise'])
        test.extend(dataset['test']['hypothesis'])

    # Remove duplicates
    train = list(set(train))
    validation = list(set(validation))
    test = list(set(test))

    train_dataset = Dataset.from_dict({'text': train})
    validation_dataset = Dataset.from_dict({'text': validation})
    test_dataset = Dataset.from_dict({'text': test})
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
    })


def get_dataloader(
    data,
    student_tokenizer,
    teacher_tokenizer,
    batch_size,
    shuffle,
    num_workers,
):
    student_data = data.map(
        lambda e: student_tokenizer(
            e['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        ),
        batched=True
    )
    student_data.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask']
    )
    teacher_data = data.map(
        lambda e: teacher_tokenizer(
            e['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        ),
        batched=True
    )
    teacher_data.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask']
    )
    dataset = torch.utils.data.TensorDataset(
        student_data['input_ids'],
        student_data['attention_mask'],
        teacher_data['input_ids'],
        teacher_data['attention_mask']
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )