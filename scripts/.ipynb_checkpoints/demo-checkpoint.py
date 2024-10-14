# Script to perform benchmarking, few-shot learning, and fine-tuning using your module.

# Import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from sklearn.model_selection import train_test_split

# Import your module classes (assuming they are defined in a module named 'model_module')
# from model_module import ModelBenchmark, ModelFewShotLearner, ModelFineTuner

# For GPT-2, we'll create subclasses to handle the GPT-2 model for each class
class ModelBenchmarkGPT2(ModelBenchmark):
    def __init__(self, model_name, X_features, y_labels, test_size=0.2, random_state=42):
        super().__init__(model_name, X_features, y_labels, test_size, random_state)

        # Adjustments for GPT-2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2ForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True
        
class ModelFewShotLearnerGPT2(ModelFewShotLearner):
    def __init__(self, model_name, X_features, y_labels, test_size=0.2, random_state=42):
        super().__init__(model_name, X_features, y_labels, test_size, random_state)

        # Adjustments for GPT-2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # GPT-2 is a left-padding model
        self.tokenizer.truncation_side = 'right'
        self.models = {}

    def few_shot_learning(self, max_N=5, batch_size=8, num_epochs=3, validation_split=0.1):
        """
        Override the few_shot_learning method to adjust for GPT-2 specifics.
        """
        results = {}

        total_samples = len(self.X_train_full)
        max_sample_size = 2 ** max_N

        # Shuffle the training data once
        combined = list(zip(self.X_train_full, self.y_train_full))
        random.shuffle(combined)
        X_train_shuffled, y_train_shuffled = zip(*combined)

        # For each sample size
        for n in range(1, max_N + 1):
            sample_size = min(2 ** n, total_samples)
            print(f"\nPerforming few-shot learning with sample size: {sample_size}")

            # Take the first 'sample_size' examples from the shuffled training data
            X_sample = X_train_shuffled[:sample_size]
            y_sample = y_train_shuffled[:sample_size]

            # Initialize a new GPT-2 model for each sample size
            model = GPT2ForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            model.resize_token_embeddings(len(self.tokenizer))
            model.config.pad_token_id = self.tokenizer.pad_token_id
            model.to(self.device)

            # Set the model to output hidden states
            model.config.output_hidden_states = True

            # Handle small sample sizes for validation
            val_size = int(validation_split * sample_size)
            if val_size < 1 or sample_size - val_size < 1:
                # If not enough data for validation, use all data for training and skip validation
                X_train = X_sample
                y_train = y_sample
                X_val = []
                y_val = []
                print(f"Not enough data for validation at sample size {sample_size}. Proceeding without validation.")
                evaluation_strategy = 'no'
                save_strategy = 'no'
                load_best_model_at_end = False
                metric_for_best_model = None
                callbacks = None
                val_dataset = None
            else:
                # Split the sample into training and validation sets
                X_train = X_sample[:sample_size - val_size]
                y_train = y_sample[:sample_size - val_size]
                X_val = X_sample[sample_size - val_size:]
                y_val = y_sample[sample_size - val_size:]
                evaluation_strategy = 'epoch'
                save_strategy = 'epoch'
                load_best_model_at_end = True
                metric_for_best_model = 'eval_loss'
                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

                # Tokenize the validation dataset
                val_encodings = self.tokenizer(
                    X_val,
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                val_dataset = self._create_dataset(val_encodings, y_val)

            # Tokenize the training dataset
            train_encodings = self.tokenizer(
                X_train,
                truncation=True,
                padding=True,
                max_length=512
            )
            train_dataset = self._create_dataset(train_encodings, y_train)

            # Define training arguments
            training_args = TrainingArguments(
                output_dir=f'./results_{sample_size}',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy=evaluation_strategy,
                save_strategy=save_strategy,
                logging_strategy='epoch',
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
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
                compute_metrics=self.compute_metrics if val_dataset else None,
                callbacks=callbacks
            )

            # Fine-tune the model
            trainer.train()

            # Save the trained model
            self.models[sample_size] = model

            # Evaluate on the same test set
            preds = self.infer(sample_size, self.X_test)
            labels = self.y_test

            # Compute performance metrics and confidence intervals
            metrics = self._compute_performance_metrics(labels, preds)

            results[sample_size] = metrics

        return results

class ModelFineTunerGPT2(ModelFineTuner):
    def __init__(self, model_name, X_features, y_labels, test_size=0.2, random_state=42):
        super().__init__(model_name, X_features, y_labels, test_size, random_state)

        # Adjustments for GPT-2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # GPT-2 is a left-padding model
        self.tokenizer.truncation_side = 'right'
        self.model = GPT2ForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.config.output_hidden_states = True

# Load the SST-2 dataset
print("Loading the SST-2 dataset...")
dataset = load_dataset('glue', 'sst2')

# Prepare the data
print("Preparing the data...")
X_train = dataset['train']['sentence']
y_train = dataset['train']['label']
X_validation = dataset['validation']['sentence']
y_validation = dataset['validation']['label']

# Combine the train and validation sets
X_features = X_train + X_validation
y_labels = y_train + y_validation

# Optionally, use a smaller subset for quicker execution (uncomment if needed)
X_features = X_features[:2000]
y_labels = y_labels[:2000]

# Function to run benchmarking
def run_benchmark(model_name, X_features, y_labels):
    print(f"\nBenchmarking model: {model_name}")
    if 'gpt2' in model_name:
        benchmark = ModelBenchmarkGPT2(model_name, X_features, y_labels)
    else:
        benchmark = ModelBenchmark(model_name, X_features, y_labels)
    metrics = benchmark.benchmark()
    print(f"\nMetrics for {model_name} (Benchmarking):")
    print(f"Accuracy: {metrics['accuracy']:.4f} with 95% CI [{metrics['accuracy_confidence_interval'][0]:.4f}, {metrics['accuracy_confidence_interval'][1]:.4f}]")
    print(f"Precision: {metrics['precision']:.4f} with 95% CI [{metrics['precision_confidence_interval'][0]:.4f}, {metrics['precision_confidence_interval'][1]:.4f}]")
    print(f"Recall: {metrics['recall']:.4f} with 95% CI [{metrics['recall_confidence_interval'][0]:.4f}, {metrics['recall_confidence_interval'][1]:.4f}]")
    print(f"F1 Score: {metrics['f1_score']:.4f} with 95% CI [{metrics['f1_confidence_interval'][0]:.4f}, {metrics['f1_confidence_interval'][1]:.4f}]")
    return metrics

# Function to run few-shot learning
def run_few_shot_learning(model_name, X_features, y_labels):
    print(f"\nFew-Shot Learning with model: {model_name}")
    if 'gpt2' in model_name:
        few_shot_learner = ModelFewShotLearnerGPT2(model_name, X_features, y_labels)
    else:
        few_shot_learner = ModelFewShotLearner(model_name, X_features, y_labels)
    results = few_shot_learner.few_shot_learning(max_N=10)  # Adjust max_N for desired sample sizes
    for sample_size, metrics in results.items():
        print(f"\nMetrics for {model_name} with sample size {sample_size} (Few-Shot Learning):")
        print(f"Accuracy: {metrics['accuracy']:.4f} with 95% CI [{metrics['accuracy_confidence_interval'][0]:.4f}, {metrics['accuracy_confidence_interval'][1]:.4f}]")
        print(f"Precision: {metrics['precision']:.4f} with 95% CI [{metrics['precision_confidence_interval'][0]:.4f}, {metrics['precision_confidence_interval'][1]:.4f}]")
        print(f"Recall: {metrics['recall']:.4f} with 95% CI [{metrics['recall_confidence_interval'][0]:.4f}, {metrics['recall_confidence_interval'][1]:.4f}]")
        print(f"F1 Score: {metrics['f1_score']:.4f} with 95% CI [{metrics['f1_confidence_interval'][0]:.4f}, {metrics['f1_confidence_interval'][1]:.4f}]")
    return results

# Function to run fine-tuning
def run_fine_tuning(model_name, X_features, y_labels):
    print(f"\nFine-Tuning model: {model_name}")
    if 'gpt2' in model_name:
        fine_tuner = ModelFineTunerGPT2(model_name, X_features, y_labels)
    else:
        fine_tuner = ModelFineTuner(model_name, X_features, y_labels)
    metrics = fine_tuner.fine_tune()
    print(f"\nMetrics for {model_name} (Fine-Tuning):")
    print(f"Accuracy: {metrics['accuracy']:.4f} with 95% CI [{metrics['accuracy_confidence_interval'][0]:.4f}, {metrics['accuracy_confidence_interval'][1]:.4f}]")
    print(f"Precision: {metrics['precision']:.4f} with 95% CI [{metrics['precision_confidence_interval'][0]:.4f}, {metrics['precision_confidence_interval'][1]:.4f}]")
    print(f"Recall: {metrics['recall']:.4f} with 95% CI [{metrics['recall_confidence_interval'][0]:.4f}, {metrics['recall_confidence_interval'][1]:.4f}]")
    print(f"F1 Score: {metrics['f1_score']:.4f} with 95% CI [{metrics['f1_confidence_interval'][0]:.4f}, {metrics['f1_confidence_interval'][1]:.4f}]")
    return metrics

# List of models to evaluate
# models = ['bert-base-uncased', 'distilbert-base-uncased', 'gpt2']
models = ['gpt2']

# Run evaluations for each model
for model_name in models:
    # Benchmarking
    # benchmark_metrics = run_benchmark(model_name, X_features, y_labels)
    
    # Few-Shot Learning
    few_shot_results = run_few_shot_learning(model_name, X_features, y_labels)
    
    # Fine-Tuning
    fine_tune_metrics = run_fine_tuning(model_name, X_features, y_labels)
