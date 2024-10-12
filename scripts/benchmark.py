"""
Module for benchmarking and fine-tuning HuggingFace models on a given dataset.

This module provides classes that can take any model from HuggingFace using AutoModel
and then provide benchmarking without changing its parameters, perform few-shot learning
experiments, and fine-tune models with early stopping and validation datasets.
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random


class ModelBenchmark:
    """
    A class for benchmarking HuggingFace models on a given dataset.

    Attributes:
        model_name (str): The name of the model to load from HuggingFace.
        X_features (List[str]): The input features (e.g., text).
        y_labels (List[int]): The true labels.
        model: The loaded model.
        tokenizer: The tokenizer associated with the model.
        device: The device to run the model on (CPU or GPU).
    """

    def __init__(self, model_name, X_features, y_labels):
        """
        Initializes the ModelBenchmark class.

        Args:
            model_name (str): The name of the model to load from HuggingFace.
            X_features (List[str]): The input features (e.g., text).
            y_labels (List[int]): The true labels.
        """
        self.model_name = model_name
        self.X_features = X_features
        self.y_labels = y_labels

        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def benchmark(self, batch_size=32):
        """
        Performs benchmarking of the model on the dataset.

        Args:
            batch_size (int): The batch size for DataLoader.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        # Tokenize the input features
        inputs = self.tokenizer(
            self.X_features,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        labels = torch.tensor(self.y_labels)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = []
        all_labels = []

        # Disable gradient calculations for inference
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate performance metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


class ModelFewShotLearner:
    """
    A class for performing few-shot learning experiments on HuggingFace models.

    This class allows you to perform few-shot learning by fine-tuning a model on small
    subsets of the dataset, where the sample sizes are powers of 2 (from 2^1 up to 2^N).

    Attributes:
        model_name (str): The name of the model to load from HuggingFace.
        X_features (List[str]): The input features (e.g., text).
        y_labels (List[int]): The true labels.
        device: The device to run the model on (CPU or GPU).
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(self, model_name, X_features, y_labels):
        """
        Initializes the ModelFewShotLearner class.

        Args:
            model_name (str): The name of the model to load from HuggingFace.
            X_features (List[str]): The input features (e.g., text).
            y_labels (List[int]): The true labels.
        """
        self.model_name = model_name
        self.X_features = X_features
        self.y_labels = y_labels

        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def few_shot_learning(self, max_N=10, batch_size=8, num_epochs=3, validation_split=0.1):
        """
        Performs few-shot learning experiments from 2^1 up to 2^N samples.

        Args:
            max_N (int): The maximum exponent N (default is 10).
            batch_size (int): The batch size for DataLoader.
            num_epochs (int): Number of epochs for fine-tuning.
            validation_split (float): Fraction of data to use for validation.

        Returns:
            dict: A dictionary with sample sizes as keys and performance metrics as values.
        """
        results = {}
        total_samples = len(self.X_features)

        # Shuffle the dataset
        combined = list(zip(self.X_features, self.y_labels))
        random.shuffle(combined)
        X_features_shuffled, y_labels_shuffled = zip(*combined)

        for n in range(1, max_N + 1):
            sample_size = min(2 ** n, total_samples)
            print(f"Performing few-shot learning with sample size: {sample_size}")

            X_sample = X_features_shuffled[:sample_size]
            y_sample = y_labels_shuffled[:sample_size]

            # Initialize a new model for each sample size to ensure independence
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(self.device)

            # Split into training and validation sets
            val_size = int(validation_split * sample_size)
            train_size = sample_size - val_size

            X_train = X_sample[:train_size]
            y_train = y_sample[:train_size]
            X_val = X_sample[train_size:]
            y_val = y_sample[train_size:]

            # Tokenize the datasets
            train_encodings = self.tokenizer(
                X_train,
                truncation=True,
                padding=True,
                max_length=512
            )
            val_encodings = self.tokenizer(
                X_val,
                truncation=True,
                padding=True,
                max_length=512
            )

            # Create torch datasets
            train_dataset = self._create_dataset(train_encodings, y_train)
            val_dataset = self._create_dataset(val_encodings, y_val)

            # Define training arguments with early stopping
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                save_total_limit=1,
                seed=42
            )

            # Define the trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Fine-tune the model
            trainer.train()

            # Evaluate on validation set
            eval_result = trainer.evaluate()

            results[sample_size] = {
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'eval_precision': eval_result.get('eval_precision', None),
                'eval_recall': eval_result.get('eval_recall', None),
                'eval_f1': eval_result.get('eval_f1', None)
            }

        return results

    def _create_dataset(self, encodings, labels):
        """
        Creates a PyTorch dataset from encodings and labels.

        Args:
            encodings (dict): Tokenized inputs.
            labels (List[int]): Corresponding labels.

        Returns:
            torch.utils.data.Dataset: A dataset object.
        """
        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return TorchDataset(encodings, labels)

    def compute_metrics(self, pred):
        """
        Compute metrics for evaluation.

        Args:
            pred: Predictions from the model.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class ModelFineTuner:
    """
    A class for fine-tuning HuggingFace models on a given dataset with early stopping and validation.

    This class allows you to fine-tune any model from HuggingFace on your dataset. It includes
    support for early stopping and uses a validation dataset to monitor performance.

    Attributes:
        model_name (str): The name of the model to load from HuggingFace.
        X_features (List[str]): The input features (e.g., text).
        y_labels (List[int]): The true labels.
        device: The device to run the model on (CPU or GPU).
        tokenizer: The tokenizer associated with the model.
        model: The loaded model.
    """

    def __init__(self, model_name, X_features, y_labels):
        """
        Initializes the ModelFineTuner class.

        Args:
            model_name (str): The name of the model to load from HuggingFace.
            X_features (List[str]): The input features (e.g., text).
            y_labels (List[int]): The true labels.
        """
        self.model_name = model_name
        self.X_features = X_features
        self.y_labels = y_labels

        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

    def fine_tune(self, batch_size=8, num_epochs=3, validation_split=0.1):
        """
        Fine-tunes the model on the dataset with early stopping and validation.

        Args:
            batch_size (int): The batch size for DataLoader.
            num_epochs (int): Number of epochs for fine-tuning.
            validation_split (float): Fraction of data to use for validation.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        total_samples = len(self.X_features)

        # Shuffle the dataset
        combined = list(zip(self.X_features, self.y_labels))
        random.shuffle(combined)
        X_features_shuffled, y_labels_shuffled = zip(*combined)

        # Split into training and validation sets
        val_size = int(validation_split * total_samples)
        train_size = total_samples - val_size

        X_train = X_features_shuffled[:train_size]
        y_train = y_labels_shuffled[:train_size]
        X_val = X_features_shuffled[train_size:]
        y_val = y_labels_shuffled[train_size:]

        # Tokenize the datasets
        train_encodings = self.tokenizer(
            X_train,
            truncation=True,
            padding=True,
            max_length=512
        )
        val_encodings = self.tokenizer(
            X_val,
            truncation=True,
            padding=True,
            max_length=512
        )

        # Create torch datasets
        train_dataset = self._create_dataset(train_encodings, y_train)
        val_dataset = self._create_dataset(val_encodings, y_val)

        # Define training arguments with early stopping
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_total_limit=1,
            seed=42
        )

        # Define the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate on validation set
        eval_result = trainer.evaluate()

        return {
            'eval_loss': eval_result['eval_loss'],
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_precision': eval_result.get('eval_precision', None),
            'eval_recall': eval_result.get('eval_recall', None),
            'eval_f1': eval_result.get('eval_f1', None)
        }

    def _create_dataset(self, encodings, labels):
        """
        Creates a PyTorch dataset from encodings and labels.

        Args:
            encodings (dict): Tokenized inputs.
            labels (List[int]): Corresponding labels.

        Returns:
            torch.utils.data.Dataset: A dataset object.
        """
        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return TorchDataset(encodings, labels)

    def compute_metrics(self, pred):
        """
        Compute metrics for evaluation.

        Args:
            pred: Predictions from the model.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
