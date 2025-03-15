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
from scipy.signal import butter, lfilter
from collections import deque
from pydub import AudioSegment
from pydub.generators import Sine


def highpass_filter(y, sr, cutoff=100):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, y)

def preprocess_audio(input_audio):
    y, sr = librosa.load(input_audio, sr=16000)
    y_filtered = highpass_filter(y,sr)
    reduced_noise = nr.reduce_noise(y=y_filtered, sr=sr,prop_decrease=1.0)
    
    # Generate cleaned file name dynamically
    file_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_wav = f"./Cleaned_audio/{file_name}_cleaned.wav"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    
    sf.write(output_wav, reduced_noise, sr)
    print("Audio Cleaned Successfully")
    return output_wav

def transcribe_audio(audio_file):
    segments, _ = model.transcribe(audio_file, language="ta", word_timestamps=True)
    print("Translated audio to tamil texts successfully")
    result = []
    #tan_texts = []
    for segment in segments:
        for word in segment.words:
            t = []
            text = word.word
            start_time = word.start
            end_time = word.end
            tanglish_text = translator(text, max_length=128)[0]['translation_text']
            tanglish_text = tanglish_text.replace(" ", "")
            # if tanglish_text not in vocab_list:
            #     tanglish_text = generate_spellings(tanglish_text,vocab_list)
            # print(tanglish_text)
            t.append(tanglish_text)
            t.append(start_time)
            t.append(end_time)
            print(text,tanglish_text)
            #tan_texts.append(tanglish_text)
            result.append(t)
            #print(tanglish_text,start_time,end_time)
    print("Translated tamil to tanglish text successfully")
    return result

# def predict_category(sample, tokenizer, preprocess_text, remove_single_characters, cv, clf):
#     sample = preprocess_text(sample)
#     sample_tokens = tokenizer.tokenizer([sample]) 
#     sample_tokens = remove_single_characters(sample_tokens[0])  
#     sample_tokens = [" ".join(sample_tokens)]  
#     data = cv.transform(sample_tokens).toarray()
#     predict = clf.predict(data)
#     return 1 if predict[0] == 'OFF' else 0

# def remove_single_characters(tokens: List[str]) -> List[str]:
#     return [token for token in tokens if len(token) > 1]

def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

suffixes_to_remove = ["da", "ga", "vaa", "ra", "la", "pa", "ma", "ta", "na","p"]

def remove_suffix(word):
    if word in vocab_list:
        #print("Remove: ",word)
        return word
    for suffix in suffixes_to_remove:
        if word.endswith(suffix):
            a  =word[:-len(suffix)]
            if a in vocab_list:
                #print("Remove: ",word)
                return word[:-len(suffix)]
    #print("Remove: ",word)
    return word

def preprocess_text(text: str) -> str:
    text = clean_text(str(text))
    for rule in custom_pre_rules:
        text = rule(text)
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
    a =  re.sub(r'(.)\1+$', r'\1', word)
    #print("Trimmed: ",a)
    return a

custom_pre_rules = [lower_case_everything, handle_all_caps, handle_upper_case_first_letter]


class CodeMixedTanglishTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
    def __call__(self, items: List[str]) -> List[List[str]]:  
        return [self.sp.EncodeAsPieces(t) for t in items]
    def tokenizer(self, items: List[str]) -> List[List[str]]:
        return [self.sp.EncodeAsPieces(t) for t in items]

input_audio_path = "./Audio/naan_oru_thadava_sonna_dialogue.mp3"
clean_audio = preprocess_audio(input_audio_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = faster_whisper.WhisperModel("medium", device=device, compute_type="float32")

sp = spm.SentencePieceProcessor()
sp.Load("Tokenizer/Tanglish/taen_spm.model")
vocab_size = sp.get_piece_size()
vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]

translator = pipeline("translation", model="ratish03/tamil_to_tanglish_model", tokenizer="ratish03/tamil_to_tanglish_model", src_lang="ta", tgt_lang="en")


def generate_spellings(word, vocab_list, max_depth=5):
    phonetic_mappings = {
        "aa": ["a"], "ae": ["e"], "ai": ["ay", "ei"], "au": ["ow", "av"], "ee": ["i", "ea", "e"], "i": ["e"],"u":["oo"],
        "oo": [ "ou", "o"], "e": ["i"], "oa": ["o"], "ua": ["wa"], "v": ["w"], "pa": ["ba"], "ka": ["ga"],
        "tha": ["ta"], "thae": ["the"], "dha": ["da"], "zh": ["l", "r"], "sh": ["s", "ch"], "u": ["oo"],"t":["d"],
        "ch": ["sh", "s"], "ph": ["f"], "j": ["z", "y"], "cho": ["sho", "so"], "chu": ["shu"], "koa": ["go"], "ko": ["go"], "bi": ["pi"],
        "che": ["she"], "ji": ["zi"], "jo": ["zo"], "ku": ["gu"], "kha": ["ka"], "ghe": ["ge"], "vai": ["vi"], "cha": ["sa"], "ga": [""], "aa": ['a']
    }

    def modify_and_check(word):
        """Generate modified versions and check if they exist in vocab_list."""
        word_parts = [[char] if char not in phonetic_mappings else [char] + phonetic_mappings[char] for char in word]
        
        for combo in product(*word_parts):
            modified_word = "".join(combo)
            if modified_word in vocab_list:
                return modified_word
        return None

    queue = deque([(word, 0)])  # (word, depth)
    visited = set([word])  # Keep track of visited words

    while queue:
        current_word, depth = queue.popleft()  # Pop from front (BFS)
        
        # Stop if we reach max depth
        if depth > max_depth:
            return word  # Return original word if no valid one found
        
        # Check if the modified word exists
        found_word = modify_and_check(current_word)
        if found_word:
            return found_word  # Found a match, return it
        
        # Generate new words
        for key in phonetic_mappings:
            if key in current_word:
                for replacement in phonetic_mappings[key]:
                    new_word = current_word.replace(key, replacement, 1)  # Replace only once per iteration
                    
                    if new_word not in visited:
                        visited.add(new_word)  # Mark as visited
                        print(new_word,depth)
                        queue.append((new_word, depth + 1))  # Add with incremented depth

    return word  # If no match found, return original word

def create_beep(duration_ms):
    return Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-5)

def beep(input_audio,timestamps):
    audio = AudioSegment.from_file(input_audio)  

    for start, end in timestamps:
        st = int(start*1000)
        en = int(end*1000)

        if en > len(audio):
            en = len(audio)
        if st >= len(audio):
            continue
        
        duration = en - st
        beep_segment = create_beep(duration)
        
        silent_segment = AudioSegment.silent(duration=duration)
        audio = audio[:st] + silent_segment + audio[en:]
        
        audio = audio.overlay(beep_segment, position=st)

    file_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_audio = f"./Censored_audio/{file_name}_censored.mp3"

    audio.export(output_audio, format="mp3")
    print(f"Beeped audio saved as {output_audio}")

result = transcribe_audio(clean_audio)
print(result)
tan_texts = []
timestamps = []

for i in result:
    stamp = []
    st_time = i[1]
    en_time = i[2]
    t = i[0]
    tan = preprocess_text(t)
    #print("preprocess: ",tan)
    tan = trim_repeated_letters(tan)
    tan = remove_suffix(tan)
    #print("Before text: ",tan)
    if tan not in vocab_list:
        tan = generate_spellings(tan,vocab_list)
    tan_texts.append(tan)

    stamp.append(st_time)
    stamp.append(en_time)
    timestamps.append(stamp)
    

bilstm_model = load_model("./Classification/bilstm_model1.h5")

tokenizer = CodeMixedTanglishTokenizer("./Tokenizer/Tanglish/taen_spm.model")

print("Tanglish",tan_texts)

hate = []
t_stamps = []
for t in range(len(tan_texts)):
    test_text = [tan_texts[t]]
    cleaned = [preprocess_text(text) for text in test_text]
    tokenized = tokenizer.tokenizer(cleaned)
    encoded = [tokenizer.sp.PieceToId(piece) for text in tokenized for piece in text]
    padded = pad_sequences([encoded],maxlen=70,padding="post")

    predictions = bilstm_model.predict(padded)
    predicted_labels = np.argmax(predictions,axis=1)

    print(cleaned,predictions,predicted_labels)

    if(predicted_labels == 1):
        hate.append(tan_texts[t])
        t_stamps.append(timestamps[t])

print("\n")
print("Hate: ",hate)
print("Timestamps: ",t_stamps)
if(len(hate) == 0):
    print("No hate speech detected")
else:
    print("Hate speech detected")

    beep(input_audio_path,t_stamps)



