model = faster_whisper.WhisperModel("medium", device=device, compute_type="float32")
segments, _ = model.transcribe(audio_file, language="ta", word_timestamps=True)

translator = pipeline("translation", model="ratish03/tamil_to_tanglish_model", tokenizer="ratish03/tamil_to_tanglish_model",
             src_lang="ta", tgt_lang="en")

for segment in segments:
    for word in segment.words:
        t = []
        text = word.word
        start_time = word.start
        end_time = word.end
        tanglish_text = translator(text, max_length=128)[0]['translation_text']



#BiLSTM
model = Sequential([
    Embedding(input_dim=max_token_id+1, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),  
    Bidirectional(LSTM(LSTM_UNITS)),  
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(label_mapping), activation="softmax")  
])

#indic-BERT
model_name = "ai4bharat/indic-bert"      
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)


meta_model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)), 
    Dense(8, activation='relu'),  
    Dense(1, activation='sigmoid')  
])


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

    