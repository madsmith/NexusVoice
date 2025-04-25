from os import PathLike
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import Tuple, Union
from pathlib import Path
import pytest

from nexusvoice.ai.classifier.intent_labels import IntentLabels

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "nexusvoice" / "models" / "distilbert-nexus"

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class IntentClassifier:
    """A wrapper around the trained classifier for easy inference"""
    
    def __init__(self, model_path: Union[str, PathLike, None] = None):
        if model_path is None:
            model_path = MODEL_DIR
        
        self.device: torch.device = get_device()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device) # type: ignore
        self.model.eval()
        self.labels = IntentLabels.all_labels()
    
    def classify(self, text: str) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            probs = torch.nn.functional.softmax(predictions, dim=1)
            prob_max, indices = torch.max(probs, dim=1)
            confidence = prob_max.item()
            predicted = int(indices.item())
            label = self.labels[predicted]
            return label, confidence

def test_intent_classifier_home_automation():
    classifier = IntentClassifier()
    label, confidence = classifier.classify("Turn on the living room lights")
    print(f"Label: {label}, Confidence: {confidence}")
    assert label == "home_automation"
    assert confidence > 0.7

def test_intent_classifier_conversation():
    classifier = IntentClassifier()
    label, confidence = classifier.classify("What's the weather like today?")
    print(f"Label: {label}, Confidence: {confidence}")
    assert label == "conversation"
    assert confidence > 0.5

