import sentencepiece as spm
from itertools import product

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("Tokenizer/Tanglish/taen_spm.model")  # Replace with your model filename

# Get the vocabulary size
vocab_size = sp.get_piece_size()

# Extract all words (subwords)
vocab_list = []
for i in range(vocab_size):
    if i=="kuthi":
        sp.id_to_piece("koothi")
    else:
        sp.id_to_piece(i)

# def generate_spellings(word, vocab_list):
#     """Generate alternative spellings of a word based on phonetic mappings and iteratively check against vocab."""
#     phonetic_mappings = {
#         "aa": ["a"], "ae": ["e"], "ai": ["ay", "ei"], "au": ["ow", "av"], "ee": ["i", "ea","e"],"i":["e"],
#         "oo": ["u", "ou","o"],"e":["i"], "oa": ["o"], "ua": ["wa"], "v": ["w"], "pa": ["ba"], "ka": ["ga"],
#         "tha": ["ta"], "thae": ["the"], "dha": ["da"], "zh": ["l", "r"], "sh": ["s", "ch"],"u":["oo"],
#         "ch": ["sh", "s"], "ph": ["f"], "j": ["z", "y"], "cho": ["sho", "so"], "chu": ["shu"],"koa":["go"],"ko":["go"],"bi":["pi"],
#         "che": ["she"], "ji": ["zi"], "jo": ["zo"], "ku": ["gu"], "kha": ["ka"], "ghe": ["ge"],"vai":["vi"],"cha":["sa"],"ga":[""],"aa":['a']
#     }

#     def modify_and_check(word):
#         word_parts = [[char] if char not in phonetic_mappings else [char] + phonetic_mappings[char] for char in word]
        
#         for combo in product(*word_parts):
#             modified_word = "".join(combo)
#             #print(modified_word)
#             if modified_word in vocab_list:
#                 return modified_word
#         return None
    
#     queue = [word]
#     visited = set()
    
#     while queue:
#         current_word = queue.pop(0)
#         if current_word in visited:
#             continue
#         visited.add(current_word)
#         found_word = modify_and_check(current_word)
#         if found_word:
#             return found_word
#         for key in phonetic_mappings:
#             if key in current_word:
#                 for replacement in phonetic_mappings[key]:
#                     new_word = current_word.replace(key, replacement, 1)
#                     queue.append(new_word)
#                     print(new_word)
    
#     return word  # If no valid match found, return original


a = 'kuthi'
print(sp.EncodeAsPieces(a))


# print(generate_spellings(a,vocab_list))


# Print vocabulary
#print(vocab_list)
