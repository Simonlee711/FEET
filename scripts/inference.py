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
from sklearn.model_selection import train_test_split


class ModelBenchmark:
    """
    A class for benchmarking HuggingFace models on a given dataset.

    Attributes:
        model_name (str): The name of the model to load from HuggingFace.
        X_train (List[str]): The input features for training (e.g., text).
        y_train (List[int]): The true labels for training.
        X_test (List[str]): The input features for testing (e.g., text).
        y_test (List[int]): The true labels for testing.
        model: The loaded model.
        tokenizer: The tokenizer associated with the model.
        device: The device to run the model on (CPU or GPU).
        embeddings (np.ndarray): The embeddings obtained during inference.
    """

    def __init__(self, model_name, X_features, y_labels, test_size=0.2, random_state=42):
        """
        Initializes the ModelBenchmark class with a train-test split.

        Args:
            model_name (str): The name of the model to load from HuggingFace.
            X_features (List[str]): The input features (e.g., text).
            y_labels (List[int]): The true labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.model_name = model_name

        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_features, y_labels, test_size=test_size, random_state=random_state
        )

        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set the model to output hidden states
        self.model.config.output_hidden_states = True

        # Initialize embeddings to None
        self.embeddings = None

    def benchmark(self, batch_size=32):
        """
        Performs benchmarking of the model on the test dataset and collects embeddings.

        Args:
            batch_size (int): The batch size for DataLoader.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        # Tokenize the test features
        inputs = self.tokenizer(
            self.X_test,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        labels = torch.tensor(self.y_test)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = []
        all_labels = []
        all_embeddings = []

        # Disable gradient calculations for inference
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                hidden_states = outputs.hidden_states  # tuple of hidden states

                # Get embeddings from the last hidden state (e.g., [CLS] token)
                embeddings = hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_size)

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                # Collect predictions, labels, and embeddings
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())

        # Calculate performance metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Store embeddings for further analysis
        self.embeddings = np.array(all_embeddings)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def get_embeddings(self):
        """
        Returns the embeddings obtained during benchmarking.

        Returns:
            np.ndarray: An array of embeddings.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Please run the benchmark method first.")
        return self.embeddings

    def infer(self, texts, batch_size=32):
        """
        Performs inference on new texts and returns predictions.

        Args:
            texts (List[str]): A list of texts to perform inference on.
            batch_size (int): Batch size for processing.

        Returns:
            List[int]: Predicted labels for the input texts.
        """
        self.model.eval()

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())

        return all_preds

    # Other methods remain the same


class ModelFewShotLearner:
    """
    A class for performing few-shot learning experiments on HuggingFace models.

    This class allows you to perform few-shot learning by fine-tuning a model on small
    subsets of the dataset, where the sample sizes are powers of 2 (from 2^1 up to 2^N).
    It also allows extraction of embeddings from the fine-tuned models and includes an
    inference function.

    Attributes:
        model_name (str): The name of the model to load from HuggingFace.
        X_train (List[str]): The input features for training (e.g., text).
        y_train (List[int]): The true labels for training.
        X_test (List[str]): The input features for testing (e.g., text).
        y_test (List[int]): The true labels for testing.
        device: The device to run the model on (CPU or GPU).
        tokenizer: The tokenizer associated with the model.
        models (dict): A dictionary to store trained models for each sample size.
    """

    def __init__(self, model_name, X_features, y_labels, test_size=0.2, random_state=42):
        """
        Initializes the ModelFewShotLearner class with a train-test split.

        Args:
            model_name (str): The name of the model to load from HuggingFace.
            X_features (List[str]): The input features (e.g., text).
            y_labels (List[int]): The true labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.model_name = model_name

        # Perform train-test split
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X_features, y_labels, test_size=test_size, random_state=random_state
        )

        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize a dictionary to store models trained with different sample sizes
        self.models = {}

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
        total_samples = len(self.X_train_full)

        # Shuffle the training data
        combined = list(zip(self.X_train_full, self.y_train_full))
        random.shuffle(combined)
        X_train_shuffled, y_train_shuffled = zip(*combined)

        for n in range(1, max_N + 1):
            sample_size = min(2 ** n, total_samples)
            print(f"Performing few-shot learning with sample size: {sample_size}")

            X_sample = X_train_shuffled[:sample_size]
            y_sample = y_train_shuffled[:sample_size]

            # Initialize a new model for each sample size
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(self.device)

            # Set the model to output hidden states
            model.config.output_hidden_states = True

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
                output_dir=f'./results_{sample_size}',
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

            # Evaluate on test set
            test_encodings = self.tokenizer(
                self.X_test,
                truncation=True,
                padding=True,
                max_length=512
            )
            test_dataset = self._create_dataset(test_encodings, self.y_test)

            eval_result = trainer.evaluate(eval_dataset=test_dataset)

            results[sample_size] = {
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'eval_precision': eval_result.get('eval_precision', None),
                'eval_recall': eval_result.get('eval_recall', None),
                'eval_f1': eval_result.get('eval_f1', None)
            }

            # Save the trained model
            self.models[sample_size] = model

        return results

    def get_embeddings(self, sample_size, texts, batch_size=32):
        """
        Generates embeddings for the provided texts using the fine-tuned model for a specific sample size.

        Args:
            sample_size (int): The sample size corresponding to the model used during training.
            texts (List[str]): A list of texts to generate embeddings for.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: An array of embeddings.

        Raises:
            ValueError: If the model for the given sample size is not found.
        """
        if sample_size not in self.models:
            raise ValueError(f"No model found for sample size {sample_size}. Please run few_shot_learning first.")

        model = self.models[sample_size]
        model.eval()

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states
                embeddings = hidden_states[-1][:, 0, :]  # [CLS] token embeddings

                all_embeddings.extend(embeddings.cpu().numpy())

        return np.array(all_embeddings)

    def infer(self, sample_size, texts, batch_size=32):
        """
        Performs inference on new texts using the model trained with a specific sample size.

        Args:
            sample_size (int): The sample size corresponding to the model used during training.
            texts (List[str]): A list of texts to perform inference on.
            batch_size (int): Batch size for processing.

        Returns:
            List[int]: Predicted labels for the input texts.

        Raises:
            ValueError: If the model for the given sample size is not found.
        """
        if sample_size not in self.models:
            raise ValueError(f"No model found for sample size {sample_size}. Please run few_shot_learning first.")

        model = self.models[sample_size]
        model.eval()

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())

        return all_preds

class ModelFineTuner:
    """
    A class for fine-tuning HuggingFace models on a given dataset with early stopping and validation.

    This class allows you to fine-tune any model from HuggingFace on your dataset. It includes
    support for early stopping and uses a validation dataset to monitor performance.
    It also provides functionality to extract embeddings from the fine-tuned model.

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

        # Set the model to output hidden states
        self.model.config.output_hidden_states = True

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

    def get_embeddings(self, texts, batch_size=32):
        """
        Generates embeddings for the provided texts using the fine-tuned model.

        Args:
            texts (List[str]): A list of texts to generate embeddings for.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: An array of embeddings.
        """
        self.model.eval()

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states
                embeddings = hidden_states[-1][:, 0, :]  # [CLS] token embeddings

                all_embeddings.extend(embeddings.cpu().numpy())

        return np.array(all_embeddings)

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
