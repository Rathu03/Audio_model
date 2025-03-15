import whisperx
import librosa
import noisereduce as nr
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import torch
import faster_whisper
from transformers import pipeline
import pickle
from typing import List
import sentencepiece as spm
import re
import string
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import product

sp = spm.SentencePieceProcessor()
sp.Load("Tokenizer/Tanglish/taen_spm.model")
vocab_size = sp.get_piece_size()
vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]

def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

def preprocess_text(text: str) -> str:
    text = clean_text(str(text))
    for rule in custom_pre_rules:
        text = rule(text)
    text = remove_suffix(text)
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

def trim_repeated_letters(word: str) -> str:
    
    a = re.sub(r'(.)\1+$', r'\1', word)
    print(a)
    return a

custom_pre_rules = [lower_case_everything, handle_all_caps, handle_upper_case_first_letter,trim_repeated_letters]

suffixes_to_remove = ["da", "ga", "vaa", "ra", "la", "pa", "ma", "ta", "na", "ya"]

def remove_suffix(word):
    if word in vocab_list:
        return word
    for suffix in suffixes_to_remove:
        if word.endswith(suffix):
            a  =word[:-len(suffix)]
            if a in vocab_list:
                return word[:-len(suffix)]
    return word

def generate_subsequences(word, min_length=3):

    if word in vocab_list:
        print(word,"is there")
        return [word]

    subsequences = set()
    n = len(word)
    
    for i in range(n):
        for j in range(i + min_length, n + 1):  
            subsequences.add(word[i:j])
    
    return list(subsequences)


class CodeMixedTanglishTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    def __call__(self, items: List[str]) -> List[List[str]]:  
        return [self.sp.EncodeAsPieces(t) for t in items]
    def tokenizer(self, items: List[str]) -> List[List[str]]:
        return [self.sp.EncodeAsPieces(t) for t in items]

bilstm_model = load_model("./Classification/bilstm_model.h5")
tokenizer = CodeMixedTanglishTokenizer("./Tokenizer/Tanglish/taen_spm.model")

# test_texts = ['ennaip', 'paarthaal', 'you', 'ellaam', 'thevidiya', 'maarith', 'theriyum', 'allavaa?']
# test_texts_cleaned = [preprocess_text(text) for text in test_texts]
# print(test_texts_cleaned)
# test_texts_tokenized = tokenizer.tokenizer(test_texts_cleaned)
# print(test_texts_tokenized)
# test_texts_encoded = [tokenizer.sp.PieceToId(piece) for text in test_texts_tokenized for piece in text]
# print(test_texts_encoded)
# test_texts_padded = pad_sequences([test_texts_encoded], maxlen=70, padding="post")
# print(test_texts_padded)


# predictions = bilstm_model.predict(test_texts_padded)
# predicted_labels = np.argmax(predictions, axis=1)
# print(predictions)
# print(predicted_labels)
test_texts = ['punda','epdi']
hate = []
for t in test_texts:
    test_text = [t]
    cleaned = [preprocess_text(text) for text in test_text]
    # print(cleaned)
    # subsequences = generate_subsequences(cleaned[0])
    # print(subsequences)
    # all_variants = set([cleaned[0]]+subsequences)
    # detected_hate = False
    # for variant in all_variants:

    tokenized = tokenizer.tokenizer(cleaned)
    encoded = [tokenizer.sp.PieceToId(piece) for text in tokenized for piece in text]
    padded = pad_sequences([encoded],maxlen=70,padding="post")

    predictions = bilstm_model.predict(padded)
    predicted_labels = np.argmax(predictions,axis=1)

    print(cleaned,predictions,predicted_labels)
#         if(predicted_labels == 1):
#             detected_hate = True
#             break
#     if detected_hate:
#         hate.append(t)
# print("Hate words detected",hate)