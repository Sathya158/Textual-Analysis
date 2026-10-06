import math
import re
import random
from collections import Counter

with open("OpenSubtitles.en-ta.en", "r", encoding="utf-8") as f:
    text = f.read()

tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
V = len(set(tokens))

unigrams = tokens
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]

uni_freq = Counter(unigrams)
bi_freq = Counter(bigrams)
tri_freq = Counter(trigrams)

# BASIC PROBABILITIES + LAPLACE
def bigram_prob(w1, w2):
    c = bi_freq.get((w1, w2), 0)
    total = uni_freq.get(w1, 0)
    return c / total if total > 0 else 0

def bigram_prob_laplace(w1, w2):
    return (bi_freq.get((w1, w2), 0) + 1) / (uni_freq.get(w1, 0) + V)

def trigram_prob(w1, w2, w3):
    c = tri_freq.get((w1, w2, w3), 0)
    total = bi_freq.get((w1, w2), 0)
    return c / total if total > 0 else 0

def trigram_prob_laplace(w1, w2, w3):
    return (tri_freq.get((w1, w2, w3), 0) + 1) / (bi_freq.get((w1, w2), 0) + V)

# BIGRAM WORD PREDICTION
def predict_next_word(w1):
    candidates = {w2: count for (a, w2), count in bi_freq.items() if a == w1}
    if not candidates:
        return None
    total = uni_freq[w1]
    probs = {w2: count / total for w2, count in candidates.items()}
    return max(probs, key=probs.get)

print("Next word after 'you':", predict_next_word("you"))
print("Next word after 'i':", predict_next_word("i"))
print("Next word after 'thank':", predict_next_word("thank"))

# TRIGRAM TEXT GENERATOR
def generate_text_trigram(w1, w2, max_words=30):
    words = [w1, w2]
    sentence_count = 0

    for _ in range(max_words):
        candidates = {w3: count for (a, b, w3), count in tri_freq.items() if a == w1 and b == w2}
        if not candidates:
            break

        next_word = random.choices(list(candidates.keys()), weights=candidates.values())[0]
        words.append(next_word)

        w1, w2 = w2, next_word

        if next_word in [".", "!", "?"]:
            sentence_count += 1
            if sentence_count >= 3:
                break

    return " ".join(words)


while True:
    user_input = input("Enter two words to start text generation: ").strip().lower()
    parts = user_input.split()
    if len(parts) == 2:
        w1, w2 = parts
        break

print(generate_text_trigram(w1, w2))


with open("OpenSubtitles.en-ta.en", "r", encoding="utf-8") as f:
    test_text = f.read(50000)

test_tokens = re.findall(r"[A-Za-z0-9]+", test_text.lower())


def unigram_prob(w):
    return (uni_freq.get(w, 0) + 1) / (sum(uni_freq.values()) + V)

def perplexity_unigram(tokens):
    log_sum = 0
    for w in tokens:
        log_sum += math.log(unigram_prob(w))
    return math.exp(-log_sum / len(tokens))


def perplexity_bigram(tokens):
    log_sum = 0
    count = 0
    for i in range(1, len(tokens)):
        log_sum += math.log(bigram_prob_laplace(tokens[i-1], tokens[i]))
        count += 1
    return math.exp(-log_sum / count)

def perplexity_trigram(tokens):
    log_sum = 0
    count = 0
    for i in range(2, len(tokens)):
        log_sum += math.log(trigram_prob_laplace(tokens[i-2], tokens[i-1], tokens[i]))
        count += 1
    return math.exp(-log_sum / count)

print("Unigram perplexity:", perplexity_unigram(test_tokens))
print("Bigram perplexity:", perplexity_bigram(test_tokens))
print("Trigram perplexity:", perplexity_trigram(test_tokens))

# KNESER–NEY SMOOTHING
D = 0.75  # discount

# Continuation counts
continuation_counts = Counter()
for (w1, w2) in bi_freq:
    continuation_counts[w2] += 1

total_continuations = sum(continuation_counts.values())

def bigram_prob_kn(w1, w2):
    c12 = bi_freq.get((w1, w2), 0)
    c1 = uni_freq.get(w1, 0)

    if c1 == 0:
        return continuation_counts[w2] / total_continuations

    p_cont = continuation_counts[w2] / total_continuations
    N1 = len([1 for (a, b) in bi_freq if a == w1])

    return max(c12 - D, 0) / c1 + (D * N1 / c1) * p_cont

def trigram_prob_kn(w1, w2, w3):
    c123 = tri_freq.get((w1, w2, w3), 0)
    c12 = bi_freq.get((w1, w2), 0)

    if c12 == 0:
        return bigram_prob_kn(w2, w3)

    N1 = len([1 for (a, b, c) in tri_freq if a == w1 and b == w2])

    return max(c123 - D, 0) / c12 + (D * N1 / c12) * bigram_prob_kn(w2, w3)

# KNESER–NEY PERPLEXITY
def perplexity_bigram_kn(tokens):
    log_sum = 0
    count = 0
    for i in range(1, len(tokens)):
        log_sum += math.log(bigram_prob_kn(tokens[i-1], tokens[i]))
        count += 1
    return math.exp(-log_sum / count)

def perplexity_trigram_kn(tokens):
    log_sum = 0
    count = 0
    for i in range(2, len(tokens)):
        log_sum += math.log(trigram_prob_kn(tokens[i-2], tokens[i-1], tokens[i]))
        count += 1
    return math.exp(-log_sum / count)

print("Bigram KN perplexity:", perplexity_bigram_kn(test_tokens))
print("Trigram KN perplexity:", perplexity_trigram_kn(test_tokens))
