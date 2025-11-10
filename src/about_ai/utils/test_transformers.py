from transformers import pipeline

classifier = pipeline("text-classification")
print(f"Classifier using '{classifier.framework}' framework.")  # should print 'pt'

# Verify datasets
from datasets import load_dataset

ds = load_dataset("imdb")
print(f"The text len is {len(ds["train"][0]["text"])} chars.")

# Verify accelerate
from accelerate import Accelerator

acc = Accelerator()
print(f"Device: {acc.device}")  # should show 'cpu' or 'cuda'

# Verify evaluate
import evaluate

metric = evaluate.load("accuracy")
print(metric.compute(references=[0, 1], predictions=[0, 1]))

# Verify tokenizers
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("bert-base-uncased")
txt = "Hello World, hello world"
print(f"Tokens in '{txt}': {tok.encode(txt).tokens}")
