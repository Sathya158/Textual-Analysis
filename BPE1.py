import random
import re
import string
from collections import Counter
from bs4 import BeautifulSoup

#Data Preparation 
with open("ta.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Clean text
text = text.lower()
text = re.sub(r'\d+', '', text)
text = text.translate(str.maketrans('', '', string.punctuation))
text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
text = BeautifulSoup(text, "html.parser").get_text()

print("Sample text:", text[:200])

words = text.split()
print("Number of words:", len(words))
print("Unique words:", len(set(words)))
print("\n20 most frequent words:", Counter(words).most_common(20))

#Word-level BPE 
# Build vocabulary: each word → char sequence + </w> 
vocab_word = Counter()
for w in words:
    seq = " ".join(list(w)) + " </w>"
    vocab_word[seq] += 1


def most_freq_pairs_word(vocab):
    pair_freqs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_freqs[pair] += freq   
    return pair_freqs


def merge_pair_word(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    merged = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, merged)
        new_vocab[new_word] = freq
    return new_vocab


print("\nWORD-LEVEL BPE (with weighted formula)")
K = 500
merge_rules_word = []
vocab = vocab_word.copy()

for i in range(K):
    pair_freqs = most_freq_pairs_word(vocab)
    if not pair_freqs:
        break
    best = max(pair_freqs, key=pair_freqs.get)
    merge_rules_word.append(best)
    vocab = merge_pair_word(best, vocab)
    print(f"Merge {i+1}: {best} -> {pair_freqs[best]}")

print("\nWord-level merge rules:")
for r in merge_rules_word:
    print(r)



#BYTE-LEVEL BPE
print("\nBYTE-LEVEL BPE")

byte_symbols = list(text)  # every character including space

def most_freq_pairs_byte(symbols_list):
    pair_freqs = Counter()
    for i in range(len(symbols_list) - 1):
        pair = (symbols_list[i], symbols_list[i+1])
        pair_freqs[pair] += 1
    return pair_freqs

def merge_pair_byte(pair, symbols_list):
    merged = "".join(pair)
    new_list = []
    i = 0
    while i < len(symbols_list):
        if i < len(symbols_list)-1 and (symbols_list[i], symbols_list[i+1]) == pair:
            new_list.append(merged)
            i += 2
        else:
            new_list.append(symbols_list[i])
            i += 1
    return new_list

K = 500
merge_rules_byte = []
symbols = byte_symbols.copy()

for i in range(K):
    pair_freqs = most_freq_pairs_byte(symbols)
    if not pair_freqs:
        break
    best = max(pair_freqs, key=pair_freqs.get)
    merge_rules_byte.append(best)
    symbols = merge_pair_byte(best, symbols)
    print(f"Merge {i+1}: {best} -> {pair_freqs[best]}")

print("\nByte-level merge rules:")
for r in merge_rules_byte:
    print(r)

#Comparison of Both Approaches
print(" COMPARISON OF WORD vs BYTE BPE")

# Word-level tokens: characters + </w>
word_level_tokens = []
for w in words:
    for ch in w:
        word_level_tokens.append(ch)
    word_level_tokens.append("</w>")

# Byte-level tokens: final merged symbols
byte_level_tokens = symbols 

#Tokenization efficiency
def tokens_per_1000_chars(tokens, text_str):
    return (len(tokens) / len(text_str)) * 1000

def tokens_per_word(tokens, words_list):
    return len(tokens) / len(words_list)

print("\n--- TOKENIZATION EFFICIENCY ---")
print("Word-level BPE:")
print("  Tokens per 1000 chars:", tokens_per_1000_chars(word_level_tokens, text))
print("  Tokens per word:", tokens_per_word(word_level_tokens, words))

print("\nByte-level BPE:")
print("  Tokens per 1000 chars:", tokens_per_1000_chars(byte_level_tokens, text))
print("  Tokens per word:", tokens_per_word(byte_level_tokens, words))


# Cross-word Tokens
def cross_word_ratio(tokens, boundary_symbol):
    cross = [t for t in tokens if boundary_symbol in t]
    return len(cross) / len(tokens)

word_cross = cross_word_ratio(word_level_tokens, "</w>")
byte_cross = cross_word_ratio(byte_level_tokens, " ")

print("\n CROSS-WORD TOKENS ")
print("Word-level BPE cross-word ratio:", word_cross)
print("Byte-level BPE cross-word ratio:", byte_cross)

# Qualitative Segmentation
def segment_word(word, merge_rules):
    tokens = list(word)
    for a, b in merge_rules:
        merged = a + b
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

long_words = [w for w in words if len(w) >= 8]
test_words = random.sample(long_words, 3)

print("\nQUALITATIVE SEGMENTATION")
for w in test_words:
    print(f"\nWord: {w}")
    print("Word-level BPE:", segment_word(w, merge_rules_word))
    print("Byte-level BPE:", segment_word(w, merge_rules_byte))
import sys 
sys.stdout = open("bpe_output.txt", "w", encoding="utf-8")