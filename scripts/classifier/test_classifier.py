from os import PathLike
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import Tuple, Union
from pathlib import Path

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
            
        # Get device
        self.device: torch.device = get_device()
        print(f"Using device: {self.device}")
            
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Move model to appropriate device
        self.model = self.model.to(self.device) # type: ignore[reportArgumentType]
        
        self.model.eval()  # Set to evaluation mode
        
        # Labels
        self.labels = IntentLabels.all_labels()
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify a piece of text and return the label and confidence"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(predictions, dim=1)
            prob_max, indices = torch.max(probs, dim=1)
            
            confidence = prob_max.item()
            predicted = int(indices.item())
            
            # Get label
            label = self.labels[predicted]
            
            return label, confidence

def main():
    # Create classifier
    print("Loading classifier...")
    classifier = IntentClassifier()
    print("Classifier loaded! Enter text to classify (or 'quit' to exit)")
    
    # Interactive loop
    while True:
        try:
            # Get input
            text = input("\nEnter text: ").strip()
            
            # Check for quit
            if text.lower() in ('quit', 'exit', 'q'):
                break
            
            # Skip empty input
            if not text:
                continue
            
            # Classify
            label, confidence = classifier.classify(text)
            
            # Print result
            print(f"\nClassification: {label}")
            print(f"Confidence: {confidence:.2%}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
