import re
from collections import Counter
import string

#1 Computation of Edit Distance
def levenshtein_distance(a, b):
    # Create a matrix (len(a)+1) x (len(b)+1)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i  
    for j in range(len(b) + 1):
        dp[0][j] = j  

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):

            if a[i - 1] == b[j - 1]:
                cost = 0  # no edit needed
            else:
                cost = 1  # substitution

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[-1][-1]

tests = [
    ("cat", "cut"),
    ("kitten", "sitting"),
    ("book", "back"),
    ("apple", "apple"),
    ("intention", "execution")
]

for w1, w2 in tests:
    print(f"{w1} → {w2} = {levenshtein_distance(w1, w2)}")


#2 Implementation of Automatic Word Correction 
def words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())

with open("w4.txt", "r", encoding="utf-8") as f:
    text = f.read()

word_list = words(text)
word_freq = Counter(word_list)

print("Total words:", len(word_list))
print("Unique words:", len(word_freq))
print("Most common words:", word_freq.most_common(10))

total_words = sum(word_freq.values())

def probability(word):
    return word_freq[word] / total_words if word in word_freq else 0

#print(probability("that"))

def edits1(word):
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    inserts = [L + c + R for L, R in splits for c in letters]
    substitutes = [L + c + R[1:] for L, R in splits if R for c in letters]
    swaps = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    return set(deletes + inserts + substitutes + swaps)

def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}
w = "world"
e1 = edits1(w)
e2 = edits2(w)

print("Edit distance 1 variants:", len(e1))
print("Edit distance 2 variants:", len(e2))


def known(words_set):
    return {w for w in words_set if w in word_freq}

def candidates(word):
    return (
        known([word]) or
        known(edits1(word)) or
        known(edits2(word)) or
        [word]
    )

def correct(word):
    return max(candidates(word), key=probability)

sentence = "I well go to the restorant tonigth and order a delicshus desert befor going home"
corrected = " ".join(correct(w) for w in words(sentence))
print(corrected)

def brute_force_correct(word):
    best_word = None
    best_distance = float("inf")

    for w in word_freq:
        d = levenshtein_distance(word, w)

        if d < best_distance:
            best_distance = d
            best_word = w
            
        elif d == best_distance:
            # tie-breaker: choose more frequent word
            if probability(w) > probability(best_word):
                best_word = w

    return best_word

print(brute_force_correct("youngre"))
print(brute_force_correct("vulnreable"))
print(brute_force_correct("advise"))
