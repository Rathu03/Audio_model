import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string

def predict_hate_speech(text, model, tokenizer, device, max_len=128):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    return {
        'text': text,
        'prediction': prediction,
        'confidence': probs[0][prediction].item(),
        'result': 'Hate Speech' if prediction == 1 else 'Not Hate Speech'
    }


model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)

model_path = "ratish03/indic-BERT-Classification"
model.load_state_dict(torch.hub.load_state_dict_from_url(f"https://huggingface.co/{model_path}/resolve/main/best_tanglish_model.pt",map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sample_texts = ['dei epdi iruka']

for text in sample_texts:
    result = predict_hate_speech(text, model, tokenizer, device)
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['result']} (Confidence: {result['confidence']:.4f})")
    print()