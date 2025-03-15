from transformers import pipeline
import sentencepiece as spm
from itertools import product

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("Tokenizer/Tanglish/taen_spm.model")

# Get vocabulary list
vocab_size = sp.get_piece_size()
vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]

# Load translation model
translator = pipeline("translation", model="ratish03/tamil_to_tanglish_model", tokenizer="ratish03/tamil_to_tanglish_model", src_lang="ta", tgt_lang="en")

def generate_spellings(word, vocab_list):
    """Generate alternative spellings of a word based on phonetic mappings and iteratively check against vocab."""
    phonetic_mappings = {
        "aa": ["a"], "ae": ["e"], "ai": ["ay", "ei"], "au": ["ow", "av"], "ee": ["i", "ea"],
        "oo": ["u", "ou","o"],"e":["i"], "oa": ["o"], "ua": ["wa"], "v": ["w"], "pa": ["ba"], "ka": ["ga"],
        "tha": ["ta"], "thae": ["the"], "dha": ["da"], "zh": ["l", "r"], "sh": ["s", "ch"],
        "ch": ["sh", "s"], "ph": ["f"], "j": ["z", "y"], "cho": ["sho", "so"], "chu": ["shu"],
        "che": ["she"], "ji": ["zi"], "jo": ["zo"], "ku": ["gu"], "kha": ["ka"], "ghe": ["ge"],"vai":["vi"],"ga":[""],"aa":['a']
    }

    def modify_and_check(word):
        word_parts = [[char] if char not in phonetic_mappings else [char] + phonetic_mappings[char] for char in word]
        
        for combo in product(*word_parts):
            modified_word = "".join(combo)
            if modified_word in vocab_list:
                return modified_word
        return None
    
    queue = [word]
    visited = set()
    
    while queue:
        current_word = queue.pop(0)
        if current_word in visited:
            continue
        visited.add(current_word)
        found_word = modify_and_check(current_word)
        if found_word:
            return found_word
        for key in phonetic_mappings:
            if key in current_word:
                for replacement in phonetic_mappings[key]:
                    new_word = current_word.replace(key, replacement, 1)
                    queue.append(new_word)
    
    return word  # If no valid match found, return original

# Process input text
tamil_text = "ஊம்புடா"
tamil_list = tamil_text.split(" ")

tanglish_list = []
for word in tamil_list:
    tanglish_text = translator(word, max_length=128)[0]['translation_text']
    tanglish_text = tanglish_text.replace(" ", "")
    
    if tanglish_text not in vocab_list:
        tanglish_text = generate_spellings(tanglish_text, vocab_list)
    
    tanglish_list.append(tanglish_text)

print("Tamil list:", tamil_list)
print("Tanglish list:", tanglish_list)


print(sp.EncodeAsPieces(tanglish_text))
