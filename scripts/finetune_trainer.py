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
        
