"""
Test script for benchmarking and fine-tuning HuggingFace models using the benchmark module.

This script demonstrates how to use the ModelBenchmark, ModelFewShotLearner, and ModelFineTuner
classes to benchmark a model, perform few-shot learning experiments, and fine-tune a model on
a sample dataset.
"""

# Import necessary modules
from benchmark_module import ModelBenchmark, ModelFewShotLearner, ModelFineTuner

# Sample dataset
X_features = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "Not bad, could be better.",
    "Absolutely fantastic!",
    "Terrible customer service.",
    "Great value for the price.",
    "I would not recommend this to my friends.",
    "Best purchase I've made this year.",
    "The quality is subpar.",
    "Exceeded my expectations!"
]

# Binary labels: 1 for positive sentiment, 0 for negative sentiment
y_labels = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

def main():
    # Specify the model name
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

    # =========================
    # Benchmarking Frozen Embeddings
    # =========================
    print("=== Benchmarking the Pre-trained Model ===")
    # Initialize the benchmark class
    benchmark = ModelBenchmark(model_name, X_features, y_labels)
    # Perform benchmarking
    benchmark_results = benchmark.benchmark()
    # Print the results
    print("Benchmarking Results:")
    for metric, value in benchmark_results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print()

    # =========================
    # Few-Shot Learning Embeddings
    # =========================
    print("=== Few-Shot Learning Experiments ===")
    # Initialize the few-shot learner class
    few_shot_learner = ModelFewShotLearner('distilbert-base-uncased', X_features, y_labels)
    # Perform few-shot learning experiments up to N=3 (sample sizes 2, 4, 8)
    few_shot_results = few_shot_learner.few_shot_learning(max_N=3, num_epochs=5)
    # Print the results
    print("Few-Shot Learning Results:")
    for sample_size, metrics in few_shot_results.items():
        print(f"Sample Size: {sample_size}")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")
        print()

    # =========================
    # Fine-Tunied Embeddings
    # =========================
    print("=== Fine-Tuning the Model ===")
    # Initialize the fine-tuner class
    fine_tuner = ModelFineTuner('distilbert-base-uncased', X_features, y_labels)
    # Fine-tune the model
    fine_tune_results = fine_tuner.fine_tune(num_epochs=5)
    # Print the results
    print("Fine-Tuning Results:")
    for metric, value in fine_tune_results.items():
        if value is not None:
            print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
    print()

if __name__ == "__main__":
    main()
