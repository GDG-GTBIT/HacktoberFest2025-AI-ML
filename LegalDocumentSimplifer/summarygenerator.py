import torch
# from transformers import (
#     BartForConditionalGeneration,
#     BartTokenizer,
# )
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from typing import Dict, List
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()


def generate_summary(text, max_length=150, min_length=50):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

text = str(input("Enter legal text:"))
print("\nGenerating summaries for test examples...")
summary = generate_summary(text)
print(summary)

