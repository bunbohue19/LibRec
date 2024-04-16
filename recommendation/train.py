import torch
import numpy as np
import pandas as pd
import csv
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from multi_label_dataset import MultiLabelDataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class MultiLabelDataset(pd.DataFrame):
    def __init__(self, 
                 df=None, 
                 tokenizer=None, 
                 max_length=1024, 
                 max_samples=None,
                 is_test=False
        ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        if max_samples is not None:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        text = self.df.loc[index, "text"]
        text = text.replace('</s>','<s>')
        labels = self.df.loc[index, "labels"]
        # process caption
        model_inputs = self.tokenizer(
            text, 
            max_length=self.max_length,  
            truncation=True,
        )
        model_inputs = {k : v[0] for k,v in model_inputs.items()}
        
        num_labels = len(label2id)
        one_hot = np.zeros(num_labels)
        for label in labels.split():
            label_id = label2id[label]
            one_hot[label_id] = 1
        
        model_inputs["labels"] = torch.tensor(one_hot, dtype=torch.float)
        
        return model_inputs

def labels_to_one_hot(label_list, label2id):
    num_labels = len(label2id)
    one_hot = np.zeros(num_labels)
    for label in label_list.split():
        label_id = label2id[label]
        one_hot[label_id] = 1
    return one_hot

def multi_label_metrics(predictions, labels, threshold=0.2):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

if __name__ == "__main__":
    # train_dataset = Dataset.from_csv('../train.csv')
    # valid_dataset = Dataset.from_csv('../val.csv')
    # test_dataset = Dataset.from_csv('../test.csv')
    
    train_dataset = pd.read_csv('../train.csv')
    valid_dataset = pd.read_csv('../val.csv')
    test_dataset = pd.read_csv('../test.csv')
    
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })
    
    labels = set()
    for label_list in dataset['train']['labels']:
        for label in label_list.split():
            labels.add(label)

    for label_list in dataset['validation']['labels']:
        for label in label_list.split():
            labels.add(label)

    for label_list in dataset['test']['labels']:
        for label in label_list.split():
            labels.add(label)

    labels = sorted(list(labels))  
    id2label = {idx : label for idx, label in enumerate(labels)}
    label2id = {label : idx for idx, label in enumerate(labels)}
        
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
    
    train_ds = MultiLabelDataset(
        df=train_dataset,
        tokenizer=tokenizer
    )
    val_ds = MultiLabelDataset(
        df=valid_dataset,
        tokenizer=tokenizer
    )
    test_ds = MultiLabelDataset(
        df=test_dataset,
        tokenizer=tokenizer
    )
    
    args = TrainingArguments(
        output_dir = '../checkpoints',
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name='bart-rec'
    )
    
    trainer = Trainer(model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()