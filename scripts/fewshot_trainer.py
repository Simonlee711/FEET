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
