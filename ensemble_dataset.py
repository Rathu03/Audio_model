import pandas as pd
import numpy as np
import torch
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
from typing import List

# Load SentencePiece Tokenizer
sp = spm.SentencePieceProcessor()
sp.Load("Tokenizer/Tanglish/taen_spm.model")

# Load BiLSTM Model
bilstm_model = load_model("./Classification/bilstm_model1.h5")

# Load BERT Model
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_path = "ratish03/indic-BERT-Classification"
model.load_state_dict(torch.hub.load_state_dict_from_url(f"https://huggingface.co/{model_path}/resolve/main/best_tanglish_model.pt", map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def remove_single_characters(tokens: List[str]) -> List[str]:
    return [token for token in tokens if len(token) > 1]

def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def lower_case_everything(t: str) -> str:
    return t.lower()

def replace_all_caps(tokens: List[str]) -> List[str]:
    return [f'xxup {t.lower()}' if t.isupper() else t for t in tokens]

def deal_caps(tokens: List[str]) -> List[str]:
    return [f'xxmaj {t}' if t.istitle() else t for t in tokens]

def handle_all_caps(t: str) -> str:
    tokens = t.split()
    tokens = replace_all_caps(tokens)
    return ' '.join(tokens)

def handle_upper_case_first_letter(t: str) -> str:
    tokens = t.split()
    tokens = deal_caps(tokens)
    return ' '.join(tokens)

custom_pre_rules = [lower_case_everything, handle_all_caps, handle_upper_case_first_letter]

def preprocess_text1(text: str) -> str:
    text = clean_text(str(text))
    for rule in custom_pre_rules:
        text = rule(text)
    return text

# Preprocessing function
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower()

# Tokenize with BiLSTM Tokenizer
class CodeMixedTanglishTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    def tokenizer(self, items):
        return [self.sp.EncodeAsPieces(t) for t in items]

tanglish_tokenizer = CodeMixedTanglishTokenizer("./Tokenizer/Tanglish/taen_spm.model")

def predict_bilstm(text):
    test_text = [text]
    cleaned_text = [preprocess_text1(t) for t in test_text]
    tokenized = tanglish_tokenizer.tokenizer(cleaned_text)
    encoded = [sp.PieceToId(piece) for text in tokenized for piece in text]
    padded = pad_sequences([encoded], maxlen=70, padding="post")
    predictions = bilstm_model.predict(padded)
    confidence = np.max(predictions)
    label = np.argmax(predictions)
    return label, confidence

def predict_bert(text, model, tokenizer, device, max_len=128):
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
    return prediction, probs[0][prediction].item()

# Load dataset
input_dataset = pd.read_csv("Dataset/Main/main_dataset.csv") 
unwanted_labels = {'OFf','label','not'}
input_dataset = input_dataset[~input_dataset['category'].isin(unwanted_labels)] 

label_mapping = {
    "NOT" : 0,
    "OFF" : 1
}
input_dataset["category"] = input_dataset["category"].map(label_mapping)


# Store results
results = []
c = 1
for index, row in input_dataset.iterrows():
    if (c % 2000 == 0):
        print("\n")
        print(f"{c} records done")
        print("\n")
    text = row['text']
    category = row['category']
    bilstm_label, bilstm_conf = predict_bilstm(text)
    bert_label, bert_conf = predict_bert(text, model, tokenizer, device)
     
    results.append({
        "text": text,
        "category": category,
        "bilstm_label": bilstm_label,
        "bilstm_confidence": bilstm_conf,
        "bert_label": bert_label,
        "bert_confidence": bert_conf
    })
    c += 1

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("hate_speech_detected.csv", index=False)
print("Results saved to hate_speech_detected.csv")
