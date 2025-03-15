import re
def trim_repeated_letters(word: str) -> str:
    a =  re.sub(r'(.)\1+$', r'\1', word)
    #print("Trimmed: ",a)
    return a

print(trim_repeated_letters("paarthalll"))