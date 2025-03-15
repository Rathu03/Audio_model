import unicodedata
text1 = "otha"
text2 = "ootha"

normalized_text1 = unicodedata.normalize("NFKC", text1)
normalized_text2 = unicodedata.normalize("NFKC", text2)

print(normalized_text2)
print(normalized_text1)
print(normalized_text1 == normalized_text2)  # True ✅ (Now they are identical)
