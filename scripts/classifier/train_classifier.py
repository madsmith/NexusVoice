from datasets import Dataset, load_from_disk, DatasetDict
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Optional
import warnings

from nexusvoice.ai.classifier.generate_training_data import generate_training_data

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "nexusvoice" / "models" / "distilbert-nexus2"
TRAINING_DIR = ROOT_DIR / "build" / "models" / "distilbert-nexus2"

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_training_data(generate_dataset: bool, save_dataset: bool = False,dataset_path: Optional[Path] = None, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Loads training and validation datasets from disk or generates them.
    - If generate_dataset is False and dataset_path is provided, attempts to load from disk.
    - If generate_dataset is True or loading fails, generates datasets and saves if path provided.
    Returns: (split_dataset ['train', 'test'])
    """
    if generate_dataset:
        texts, labels = generate_training_data()

        # Convert into a dataset
        dataset = Dataset.from_dict({"text": texts, "label": labels})

        # Create train/test split
        split_dataset = dataset.train_test_split(
            shuffle=True,
            keep_in_memory=True,
            test_size=test_size,
            seed=seed
        )

        # Save datasets if path provided
        if save_dataset:
            if not dataset_path:
                raise ValueError("dataset_path must be provided when save_dataset is True")
            split_dataset.save_to_disk(dataset_path)

        return split_dataset

    # We are not generating a dataset, so we try to load from disk
    if not dataset_path:
        raise ValueError("dataset_path must be provided when generate_dataset is False")
    
    loaded = load_from_disk(dataset_path)
    if isinstance(loaded, DatasetDict):
        if "train" not in loaded or "test" not in loaded:
            raise ValueError("DatasetDict must contain 'train' and 'test' splits!")
        return loaded
    elif isinstance(loaded, Dataset):
        # Dataset wasn't split, so we split it
        split = loaded.train_test_split(test_size=test_size, seed=seed)
        if save_dataset:
            split_path = dataset_path / "split"
            logger.warning(f"Loaded dataset is not split, saving split dataset to disk at new path: {split_path}")
            split.save_to_disk(split_path)
        return split

    raise ValueError(f"Unknown dataset type: {type(loaded)}")

def load_tokenized_dataset(dataset: DatasetDict, tokenize_function, dataset_path: Path | None =None) -> DatasetDict:
    # Attempt to load from disk
    if dataset_path:
        tokenized_dataset_path = dataset_path / "tokenized"
        if tokenized_dataset_path.exists():
            tokenized_data = load_from_disk(tokenized_dataset_path)
            if isinstance(tokenized_data, DatasetDict):
                return tokenized_data
            else:
                raise ValueError(f"Unknown tokenized dataset type: {type(tokenized_data)}")

    # Tokenize datasets
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Save tokenized datasets if path provided
    if dataset_path:
        tokenized_dataset_path = dataset_path / "tokenized"
        tokenized_dataset.save_to_disk(tokenized_dataset_path)

    return tokenized_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='weighted',
        zero_division=1.0  # type: ignore[reportArgumentType]
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Train classifier with optional dataset caching.")
    parser.add_argument('--dataset', type=Path, default=None, help='Path to directory containing train_dataset and test_dataset')
    parser.add_argument('--save-dataset', "--save",action='store_true', default=False, help='Save raw and tokenized datasets')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion for validation split if generating dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--evals-per-epoch', type=int, default=5, help='Number of evaluation steps per epoch')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of data to use for training and evaluation (0 < sample-rate <= 1)')
    args = parser.parse_args()

    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    print("Loading dataset...")
    try:
        # Try to load dataset from disk, fallback to generation if needed
        dataset = load_training_data(
            generate_dataset=False,
            save_dataset=args.save_dataset,
            dataset_path=args.dataset,
            test_size=args.test_size,
            seed=args.seed
        )
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}. Generating new dataset.")
        dataset = load_training_data(
            generate_dataset=True,
            save_dataset=args.save_dataset,
            dataset_path=args.dataset,
            test_size=args.test_size,
            seed=args.seed
        )

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    model = model.to(device) # type: ignore[reportArgumentType]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Tokenize datasets
    print("Loading tokenized datasets...")
    tokenized_dataset = load_tokenized_dataset(dataset, tokenize_function, args.dataset)
    
    # Subsample the tokenized dataset if sample_rate < 1.0
    if args.sample_rate < 1.0:
        def subsample(ds, rate):
            n = int(len(ds) * rate)
            indices = np.random.choice(len(ds), n, replace=False)
            return ds.select(indices.tolist())
        
        tokenized_dataset["train"] = subsample(tokenized_dataset["train"], args.sample_rate)
        tokenized_dataset["test"] = subsample(tokenized_dataset["test"], args.sample_rate)
        print(f"Subsampled dataset to {len(tokenized_dataset['train'])} train and {len(tokenized_dataset['test'])} test examples.")

    # Create training arguments
    train_size = len(tokenized_dataset["train"])
    batch_size = 16  # or get from args
    steps_per_epoch = int(np.ceil(train_size / batch_size))
    evals_per_epoch = args.evals_per_epoch

    eval_steps = max(1, steps_per_epoch // evals_per_epoch)
    save_steps = eval_steps

    print(f"Steps per epoch: {steps_per_epoch} - Evals per epoch: {evals_per_epoch} - Eval steps: {eval_steps}")
    training_args = TrainingArguments(
        output_dir=str(TRAINING_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(TRAINING_DIR / 'logs'),
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Enable GPU training
        no_cuda=False,
        use_mps_device=str(device) == "mps"
    )
    
    # Create trainer
    # Note: tokenizer param is deprecated and tokenizer should be specified as a processing class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    # Save model
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

if __name__ == "__main__":
    main()
