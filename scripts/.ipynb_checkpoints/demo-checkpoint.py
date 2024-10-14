import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from sklearn.model_selection import train_test_split
from model_module import ModelBenchmark, ModelFewShotLearner, ModelFineTuner

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
models = ['bert-base-uncased', 'distilbert-base-uncased', 'gpt2']

# Run evaluations for each model
for model_name in models:
    # Benchmarking
    benchmark_metrics = run_benchmark(model_name, X_features, y_labels)
    
    # Few-Shot Learning
    few_shot_results = run_few_shot_learning(model_name, X_features, y_labels)
    
    # Fine-Tuning
    fine_tune_metrics = run_fine_tuning(model_name, X_features, y_labels)
