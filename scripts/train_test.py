import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from lightgbm import LGBMClassifier

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
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from tqdm import tqdm

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

def evaluate_antibiotics_with_confidence_intervals(X_train, X_test, train, test, antibiotics, n_bootstraps=1000):
    """
    Function to train and evaluate a model for each antibiotic in the list, including confidence intervals
    for metrics using bootstrapping.

    Parameters:
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - train: Training dataset containing the targets
    - test: Testing dataset containing the targets
    - antibiotics: List of antibiotics to evaluate
    - n_bootstraps: Number of bootstrap samples to use for confidence intervals

    Returns:
    - A dictionary containing evaluation results and confidence intervals for each antibiotic.
    """
    results = {}
    for antibiotic in tqdm(antibiotics,desc="Iterating through Antibiotics Progress: "):
        y_train = train[antibiotic].astype(int).reset_index(drop=True)
        y_test = test[antibiotic].astype(int).reset_index(drop=True)
        
        # Initialize and fit the model
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
        model.fit(X_train, y_train)
        
        # Predict on test set and calculate probabilities
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Initial evaluation
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)
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
        roc_aucs = []
        prc_aucs = []
        f1_scores_list = []

        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]

            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        # Store results including confidence intervals
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

def evaluate_antibiotics_with_confidence_intervals_trainer(X_train_texts, X_test_texts, train, test, antibiotics, model_name, freeze_model=False, n_bootstraps=1000):
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
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            load_best_model_at_end=True,
            fp16=True,
        )

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
        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]
            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        results[antibiotic] = {
            'Optimal Threshold': metrics['optimal_threshold'],
            'Test Metrics': {
                'F1 Score': metrics['f1'],
                'Matthews Correlation Coefficient': metrics['mcc'],
                'ROC AUC': metrics['roc_auc'],
                'PRC AUC': metrics['prc_auc'],
                'fpr': metrics['fpr'],
                'tpr': metrics['tpr'],
                'auprc': metrics['auprc'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
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



