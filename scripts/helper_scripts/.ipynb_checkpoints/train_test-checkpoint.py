import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from lightgbm import LGBMClassifier
from transformers import (
    AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, 
    DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline, 
    AdamW, get_scheduler, pipeline, RobertaTokenizerFast
)
#from peft import LoraConfig, get_peft_model,  TaskType
from scripts.encoder import encode_texts, encode_texts_biolm

def evaluate_antibiotics(X_train, X_test, train, test, antibiotics):
    """
    Function to train and evaluate a model for each antibiotic in the list.

    Parameters:
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - train: Training dataset containing the targets
    - test: Testing dataset containing the targets
    - antibiotics: List of antibiotics to evaluate

    Returns:
    - A dictionary containing evaluation results for each antibiotic.
    """
    results = {}
    for antibiotic in antibiotics:
        print(f"Evaluating: {antibiotic}")
        y_train = train[antibiotic].astype(int)
        y_test = test[antibiotic].astype(int)
        
        # Initialize and fit the model
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
        model.fit(X_train, y_train)
        
        # Predict on test set and calculate probabilities
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Evaluate the model
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_proba)
        prc_auc_test = average_precision_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auprc = auc(recall, precision)
        
        # Store results
        results[antibiotic] = {
            'Optimal Threshold': optimal_threshold,
            'Test Metrics': {
                'F1 Score': optimal_f1,
                'Matthews Correlation Coefficient': mcc_test,
                'ROC AUC': roc_auc_test,
                'PRC AUC': prc_auc_test,
                'fpr': fpr,
                'tpr': tpr,
                'auprc': auprc,
                'precision': precision,
                'recall': recall
            }
        }
    return results

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (precision_recall_curve, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, auc)
from sklearn.utils import resample
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class AntibioticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch[0].items()}  # Input tensors
            labels = batch[1].to(device)  # Labels tensor

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

def evaluate_antibiotics_with_confidence_intervals(X_train_texts, X_test_texts, train, test, antibiotics, model_name, n_bootstraps=1000):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for antibiotic in tqdm(antibiotics, desc="Processing antibiotics"):
        print(f"Fine-tuning and evaluating for {antibiotic}")
        
        # Prepare data
        y_train = train[antibiotic].astype(int).values
        y_test = test[antibiotic].astype(int).values

        # Create datasets and dataloaders
        full_train_dataset = AntibioticDataset(X_train_texts, y_train, tokenizer)
        test_dataset = AntibioticDataset(X_test_texts, y_test, tokenizer)
        
        # Split training data into train and validation
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        # Training loop with early stopping
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        num_epochs = 10
        patience = 3
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in tqdm(range(num_epochs), desc="Training"):
            tqdm.write(f"Epoch {epoch+1}/{num_epochs}")
            model.train()
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                labels = batch[1].to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate on validation set
            val_loss, val_accuracy = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model
                torch.save(model.state_dict(), f'best_model_{antibiotic}.pt')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Load the best model for evaluation
        model.load_state_dict(torch.load(f'best_model_{antibiotic}.pt'))

        # Evaluation
        model.eval()
        y_test_proba = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                outputs = model(**inputs)
                y_test_proba.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy())

        y_test_proba = np.array(y_test_proba)

        # Calculate metrics
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_proba)
        prc_auc_test = average_precision_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auprc = auc(recall, precision)

        # Bootstrap confidence intervals
        roc_aucs, prc_aucs, f1_scores_list = [], [], []
        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]
            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        # Store results
        results[antibiotic] = {
            'Optimal Threshold': optimal_threshold,
            'Test Metrics': {
                'F1 Score': optimal_f1,
                'Matthews Correlation Coefficient': mcc_test,
                'ROC AUC': roc_auc_test,
                'PRC AUC': prc_auc_test,
                'fpr': fpr,
                'tpr': tpr,
                'auprc': auprc,
                'precision': precision,
                'recall': recall
            },
            'Confidence Intervals': {
                'ROC AUC': {'Mean': np.mean(roc_aucs), '95% CI': np.percentile(roc_aucs, [2.5, 97.5])},
                'PRC AUC': {'Mean': np.mean(prc_aucs), '95% CI': np.percentile(prc_aucs, [2.5, 97.5])},
                'F1 Score': {'Mean': np.mean(f1_scores_list), '95% CI': np.percentile(f1_scores_list, [2.5, 97.5])}
            }
        }

    return results

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, thresholds = precision_recall_curve(labels, pred.predictions[:, 1])
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else None
    optimal_f1 = f1_scores[optimal_idx]
    mcc = matthews_corrcoef(labels, preds)
    roc_auc = roc_auc_score(labels, pred.predictions[:, 1])
    prc_auc = average_precision_score(labels, pred.predictions[:, 1])
    fpr, tpr, _ = roc_curve(labels, pred.predictions[:, 1])
    auprc = auc(recall, precision)
    return {
        'f1': optimal_f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'auprc': auprc,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'optimal_threshold': optimal_threshold
    }

from torch.utils.data import random_split

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import resample
import torch
import numpy as np

def evaluate_antibiotics_with_confidence_intervals2(X_train_texts, X_test_texts, train, test, antibiotics, model_name, freeze_model=False, n_bootstraps=1000):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for antibiotic in tqdm(antibiotics, desc="Processing antibiotics"):
        print(f"Fine-tuning and evaluating for {antibiotic}")
        
        # Prepare data
        y_train = train[antibiotic].astype(int).values
        y_test = test[antibiotic].astype(int).values

        # Create datasets and dataloaders
        #from your_dataset_class import AntibioticDataset  # Replace 'your_dataset_class' with your actual dataset class file
        full_train_dataset = AntibioticDataset(X_train_texts, y_train, tokenizer)
        test_dataset = AntibioticDataset(X_test_texts, y_test, tokenizer)
        
        # Split training data into train and validation
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        if freeze_model:
            for param in model.bert.parameters():
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir='./results',
            logging_dir='./logs',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=5,
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            load_best_model_at_end=True,
            fp16=True,
        )

        # Trainer setup
        def compute_metrics(preds):
            # Define how to compute metrics here
            return {}  # Return a dictionary of metrics
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()

        preds = trainer.predict(test_dataset)
        metrics = compute_metrics(preds)

        y_test_proba = preds.predictions[:, 1]

        roc_aucs, prc_aucs, f1_scores_list = [], [], []
        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping", leave=False):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]
            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        results[antibiotic] = {
            'Optimal Threshold': metrics.get('optimal_threshold', None),
            'Test Metrics': {
                'F1 Score': metrics.get('f1', None),
                'Matthews Correlation Coefficient': metrics.get('mcc', None),
                'ROC AUC': metrics.get('roc_auc', None),
                'PRC AUC': metrics.get('prc_auc', None),
                'fpr': metrics.get('fpr', None),
                'tpr': metrics.get('tpr', None),
                'auprc': metrics.get('auprc', None),
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None)
            },
            'Confidence Intervals': {
                'ROC AUC': {'Mean': np.mean(roc_aucs), '95% CI': np.percentile(roc_aucs, [2.5, 97.5])},
                'PRC AUC': {'Mean': np.mean(prc_aucs), '95% CI': np.percentile(prc_aucs, [2.5, 97.5])},
                'F1 Score': {'Mean': np.mean(f1_scores_list), '95% CI': np.percentile(f1_scores_list, [2.5, 97.5])}
            }
        }

    return results


def print_results(results):
    # Print results
    for antibiotic, res in results.items():
        print(f"Results for {antibiotic}:")

        # Calculate mean and confidence interval half-width for F1 score
        f1_mean = res['Confidence Intervals']['F1 Score']['Mean']
        f1_ci_lower = res['Confidence Intervals']['F1 Score']['95% CI'][0]
        f1_ci_upper = res['Confidence Intervals']['F1 Score']['95% CI'][1]
        f1_error = (f1_ci_upper - f1_ci_lower) / 2

        # Calculate mean and confidence interval half-width for ROC AUC
        roc_auc_mean = res['Confidence Intervals']['ROC AUC']['Mean']
        roc_auc_ci_lower = res['Confidence Intervals']['ROC AUC']['95% CI'][0]
        roc_auc_ci_upper = res['Confidence Intervals']['ROC AUC']['95% CI'][1]
        roc_auc_error = (roc_auc_ci_upper - roc_auc_ci_lower) / 2

        # Calculate mean and confidence interval half-width for PRC AUC
        prc_auc_mean = res['Confidence Intervals']['PRC AUC']['Mean']
        prc_auc_ci_lower = res['Confidence Intervals']['PRC AUC']['95% CI'][0]
        prc_auc_ci_upper = res['Confidence Intervals']['PRC AUC']['95% CI'][1]
        prc_auc_error = (prc_auc_ci_upper - prc_auc_ci_lower) / 2

        # Print the metrics with confidence intervals
        print(f"  Test - F1: {f1_mean:.4f} +/- {f1_error:.4f}, MCC: {res['Test Metrics']['Matthews Correlation Coefficient']:.4f}, "
              f"ROC-AUC: {roc_auc_mean:.4f} +/- {roc_auc_error:.4f}, PRC-AUC: {prc_auc_mean:.4f} +/- {prc_auc_error:.4f}")
        
        
from torch.utils.data import random_split, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import torch
import numpy as np

def few_shot_learning(X_train_texts, X_test_texts, train, test, antibiotics, model_name, n_shots_list, freeze_model=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for antibiotic in antibiotics:
        print(f"Processing for antibiotic: {antibiotic}")
        
        y_train = train[antibiotic].astype(int).values
        y_test = test[antibiotic].astype(int).values
        full_train_dataset = AntibioticDataset(X_train_texts, y_train, tokenizer)
        test_dataset = AntibioticDataset(X_test_texts, y_test, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=64)

        for n_shots in n_shots_list:
            if len(full_train_dataset) < n_shots:
                print(f"Not enough data for {n_shots} shots for {antibiotic}.")
                continue

            subset_indices = np.random.choice(len(full_train_dataset), n_shots, replace=False)
            few_shot_dataset = Subset(full_train_dataset, subset_indices)
            train_loader = DataLoader(few_shot_dataset, batch_size=16, shuffle=True)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            if freeze_model:
                for param in model.base_model.parameters():
                    param.requires_grad = False

            training_args = TrainingArguments(
                output_dir=f'./results/{antibiotic}_{n_shots}_shots',
                evaluation_strategy='no',
                num_train_epochs=3,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                load_best_model_at_end=False,
                no_cuda=not torch.cuda.is_available(),
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=few_shot_dataset
            )

            trainer.train()

            # Manual evaluation
            # Manual evaluation
            model.eval()
            final_metrics = {'f1': [], 'auroc': [], 'auprc': []}
            with torch.no_grad():
                for batch in test_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits

                    predictions = torch.argmax(logits, dim=1)
                    # Move predictions and labels to CPU for metric calculation
                    predictions = predictions.cpu()
                    labels = labels.cpu()

                    softmax_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Move to CPU and convert to numpy

                    # Calculate F1 score, AUROC and AUPRC using CPU tensors or numpy arrays
                    final_metrics['f1'].append(f1_score(labels.numpy(), predictions.numpy()))
                    final_metrics['auroc'].append(roc_auc_score(labels.numpy(), softmax_probs))
                    precision, recall, _ = precision_recall_curve(labels.numpy(), softmax_probs)
                    final_metrics['auprc'].append(auc(recall, precision))

            eval_result = {key: np.mean(vals) for key, vals in final_metrics.items()}
            print(f"Results for {antibiotic} with {n_shots} shots: {eval_result}")


            results[(antibiotic, n_shots)] = eval_result

    return results

from torch.utils.data import Dataset

class AntibioticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels = labels
        self.tokenizer = tokenizer
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # Ensure no memory leak
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

