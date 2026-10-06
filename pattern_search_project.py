"""
Pattern Searching Algorithms – Full Analysis
============================================
Covers:
  1. Implementations: Brute Force, KMP, Boyer-Moore-Horspool
  2. Testing on Short / Long / DNA texts with 3 patterns each
  3. Comparison table of character comparisons
  4. Visualisation charts
"""

import random
import string
import textwrap
from dataclasses import dataclass
from typing import List


# ═══════════════════════════════════════════════════════════════
# PART 1 – Algorithm implementations
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    algorithm:   str
    pattern:     str
    positions:   List[int]
    comparisons: int


# ── Brute Force ─────────────────────────────────────────────────

def brute_force(text: str, pattern: str) -> SearchResult:
    """
    Naive O(n·m) search.
    Aligns pattern at every position and compares left-to-right.
    On any mismatch: slide one position right, restart from j=0.

    Best : O(n)     Worst : O(n·m)     Space : O(1)
    """
    n, m = len(text), len(pattern)
    positions, comparisons = [], 0

    for i in range(n - m + 1):
        for j in range(m):
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
        else:
            positions.append(i)

    return SearchResult("Brute Force", pattern, positions, comparisons)


# ── KMP ─────────────────────────────────────────────────────────

def _build_failure(pattern: str) -> List[int]:
    """
    KMP failure (partial-match) table.
    failure[i] = length of longest proper prefix of pattern[0..i]
                 that is also a suffix.
    Used to skip re-comparisons after a mismatch.
    """
    m, failure, k = len(pattern), [0] * len(pattern), 0
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = failure[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        failure[i] = k
    return failure


def kmp(text: str, pattern: str) -> SearchResult:
    """
    Knuth-Morris-Pratt – guaranteed O(n + m).
    Uses the failure function to jump j on mismatch instead of
    resetting to 0, saving redundant comparisons on repeated prefixes.

    Best : O(n)   Worst : O(n+m)   Space : O(m)
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return SearchResult("KMP", pattern, [], 0)

    failure = _build_failure(pattern)
    positions, comparisons, j = [], 0, 0

    for i in range(n):
        comparisons += 1
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]
            comparisons += 1
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = failure[j - 1]

    return SearchResult("KMP", pattern, positions, comparisons)


# ── Boyer-Moore-Horspool ─────────────────────────────────────────

def _build_bch_table(pattern: str) -> dict:
    """
    Bad-character shift table for BMH.
    shift[c] = how far to move the window when c is the rightmost
               text character and there is a mismatch.
    Characters absent from the pattern get the default shift = m.
    The last pattern character is excluded (shift of 0 → infinite loop).
    """
    m = len(pattern)
    table = {}
    for i in range(m - 1):
        table[pattern[i]] = m - 1 - i
    return table


def bmh(text: str, pattern: str) -> SearchResult:
    """
    Boyer-Moore-Horspool – avg O(n/m), best O(n/m), worst O(n·m).
    Compares right-to-left inside each window; on any mismatch looks
    up the rightmost text character to compute a (often large) skip.
    Excels on large alphabets and medium-to-long patterns.

    Best : O(n/m)   Worst : O(n·m)   Space : O(σ)
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return SearchResult("BMH", pattern, [], 0)

    table = _build_bch_table(pattern)
    positions, comparisons, i = [], 0, 0

    while i <= n - m:
        j = m - 1
        comparisons += 1
        while j > 0 and pattern[j] == text[i + j]:
            j -= 1
            comparisons += 1
        if pattern[j] == text[i + j] and j == 0:
            positions.append(i)
        i += table.get(text[i + m - 1], m)

    return SearchResult("BMH", pattern, positions, comparisons)


# ═══════════════════════════════════════════════════════════════
# PART 2 – Test data
# ═══════════════════════════════════════════════════════════════

SHORT_TEXT = (
    "the quick brown fox jumps over the lazy dog and the fox ran "
    "away quickly from the brown lazy dog today near the old fox den"
)  # ~100 chars

LONG_TEXT = """
In the realm of computer science, algorithms for searching patterns within text 
are among the most fundamental and widely applied techniques. From the simplest 
brute-force approach to the sophisticated Boyer-Moore family, each algorithm 
offers a unique balance of preprocessing cost and search efficiency. The 
brute-force algorithm compares the pattern against every possible position in 
the text, making it straightforward to understand but potentially slow for large 
inputs. The Knuth-Morris-Pratt algorithm improves on this by precomputing a 
failure function that captures the self-similarity of the pattern, allowing the 
search to skip ahead rather than restarting from scratch after a mismatch. The 
Boyer-Moore-Horspool algorithm, a simplified variant of the full Boyer-Moore 
method, achieves sub-linear average-case performance on natural language by 
shifting the pattern window by large amounts when a bad character is encountered. 
Understanding when each algorithm excels requires analysing the structure of both 
the text and the pattern: repetitive patterns favour KMP, large alphabets favour 
BMH, and very short patterns often make brute force competitive due to its lack 
of preprocessing overhead. This analysis tests all three algorithms across 
diverse inputs to quantify these trade-offs precisely.
""".replace("\n", " ").strip()  # ~1000 chars

random.seed(42)
DNA_TEXT = "".join(random.choices("ACGT", k=400))

# Patterns per dataset
SHORT_PATTERNS = ["fox",  "lazy dog", "brown fox jumps"]
LONG_PATTERNS  = ["algorithm", "Boyer-Moore", "preprocessing overhead"]
DNA_PATTERNS   = ["ACGT", "GATTACA", "ACGTACGT"]

DATASETS = [
    ("Short text (~100 chars)",    SHORT_TEXT,  SHORT_PATTERNS),
    ("Long text (~1000 chars)",    LONG_TEXT,   LONG_PATTERNS),
    ("DNA sequence (~400 chars)",  DNA_TEXT,    DNA_PATTERNS),
]

ALGORITHMS = [brute_force, kmp, bmh]


# ═══════════════════════════════════════════════════════════════
# PART 3 – Run tests and collect results
# ═══════════════════════════════════════════════════════════════

rows = []   # (dataset_label, pattern, bf_comps, kmp_comps, bmh_comps, matches)

print("\n" + "═" * 70)
print("  PATTERN SEARCH – FULL TEST RESULTS")
print("═" * 70)

for ds_label, text, patterns in DATASETS:
    print(f"\n{'─'*70}")
    print(f"  Dataset : {ds_label}  (len={len(text)})")
    print(f"{'─'*70}")

    for pattern in patterns:
        results = {fn.__name__: fn(text, pattern) for fn in ALGORITHMS}
        bf  = results["brute_force"]
        k   = results["kmp"]
        b   = results["bmh"]

        assert bf.positions == k.positions == b.positions, \
            f"Position mismatch! BF={bf.positions} KMP={k.positions} BMH={b.positions}"

        rows.append((ds_label, pattern, bf.comparisons, k.comparisons,
                     b.comparisons, len(bf.positions)))

        print(f"\n  Pattern : '{pattern}'  →  {len(bf.positions)} match(es) at {bf.positions}")
        print(f"  {'Algorithm':<18} {'Comparisons':>12}  {'Positions'}")
        print(f"  {'─'*50}")
        for r in [bf, k, b]:
            print(f"  {r.algorithm:<18} {r.comparisons:>12}  {r.positions}")


# ═══════════════════════════════════════════════════════════════
# PART 3 – Pretty comparison table
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "═" * 78)
print("  CHARACTER COMPARISONS TABLE")
print("═" * 78)
print(f"  {'Dataset':<30} {'Pattern':<25} {'BF':>6} {'KMP':>6} {'BMH':>6} {'Matches':>7}")
print(f"  {'─'*70}")

for row in rows:
    ds, pat, bf_c, kmp_c, bmh_c, m = row
    # Shorten dataset label for table
    ds_short = ds.split("(")[0].strip()
    print(f"  {ds_short:<30} {pat:<25} {bf_c:>6} {kmp_c:>6} {bmh_c:>6} {m:>7}")

print(f"  {'─'*70}")
print()


# ═══════════════════════════════════════════════════════════════
# PART 4 – Visualisation
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {"Brute Force": "#EF9F27", "KMP": "#378ADD", "BMH": "#1D9E75"}
ALGO_LABELS = ["Brute Force", "KMP", "BMH"]

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#FAFAF8")

# ── Subplot layout ───────────────────────────────────────────────
# Row 1: grouped bar chart for all 9 test cases
# Row 2: line chart comparisons vs text length  |  bar chart: BF worst-case scaling
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 2, 3)
ax3 = fig.add_subplot(3, 2, 4)
ax4 = fig.add_subplot(3, 2, 5)
ax5 = fig.add_subplot(3, 2, 6)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#FAFAF8")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#2C2C2A")
    ax.set_xlabel(xlabel, fontsize=9, color="#5F5E5A")
    ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")
    ax.tick_params(colors="#888780", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

# ── Chart 1: grouped bars for all 9 test cases ──────────────────
labels_short = [f"'{p}'" for p in SHORT_PATTERNS + LONG_PATTERNS + DNA_PATTERNS]
bf_vals  = [r[2] for r in rows]
kmp_vals = [r[3] for r in rows]
bmh_vals = [r[4] for r in rows]

x = np.arange(len(rows))
w = 0.25
ax1.bar(x - w, bf_vals,  w, color=COLORS["Brute Force"], label="Brute Force", zorder=3)
ax1.bar(x,     kmp_vals, w, color=COLORS["KMP"],         label="KMP",         zorder=3)
ax1.bar(x + w, bmh_vals, w, color=COLORS["BMH"],         label="BMH",         zorder=3)

# Dataset separators
ax1.axvline(2.5, color="#B4B2A9", linewidth=1, linestyle=":")
ax1.axvline(5.5, color="#B4B2A9", linewidth=1, linestyle=":")

# Dataset labels above bars
for xpos, lbl in [(1, "Short text"), (4, "Long text"), (7, "DNA sequence")]:
    ax1.text(xpos, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
             lbl, ha="center", va="bottom", fontsize=8, color="#5F5E5A",
             style="italic")

ax1.set_xticks(x)
short_labels = [textwrap.shorten(l, 18) for l in labels_short]
ax1.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7.5)
style_ax(ax1, "Character comparisons – all test cases", "Pattern", "Comparisons")
ax1.legend(fontsize=8, framealpha=0.6)

# Fix y-lim so dataset labels don't get clipped
ax1.set_ylim(0, max(max(bf_vals), max(kmp_vals), max(bmh_vals)) * 1.18)
# Re-draw dataset labels now that ylim is fixed
for xpos, lbl in [(1, "Short text"), (4, "Long text"), (7, "DNA sequence")]:
    ax1.text(xpos, ax1.get_ylim()[1] * 0.97,
             lbl, ha="center", va="top", fontsize=8, color="#5F5E5A", style="italic")

# ── Chart 2: comparisons vs text length (fixed pattern) ─────────
pattern_fixed = "ACGT"
text_lengths  = [50, 100, 200, 400, 800, 1600]
bf_l, kmp_l, bmh_l = [], [], []
for tl in text_lengths:
    t = "".join(random.choices("ACGT", k=tl))
    bf_l.append(brute_force(t, pattern_fixed).comparisons)
    kmp_l.append(kmp(t, pattern_fixed).comparisons)
    bmh_l.append(bmh(t, pattern_fixed).comparisons)

ax2.plot(text_lengths, bf_l,  "o-", color=COLORS["Brute Force"], linewidth=1.5, markersize=5, label="Brute Force")
ax2.plot(text_lengths, kmp_l, "s-", color=COLORS["KMP"],         linewidth=1.5, markersize=5, label="KMP")
ax2.plot(text_lengths, bmh_l, "^-", color=COLORS["BMH"],         linewidth=1.5, markersize=5, label="BMH")
style_ax(ax2, f"Comparisons vs text length  (pattern='{pattern_fixed}')", "Text length (chars)", "Comparisons")
ax2.legend(fontsize=7)

# ── Chart 3: comparisons vs pattern length (fixed text) ─────────
text_fixed   = "".join(random.choices("ACGT", k=500))
pat_lengths  = [2, 4, 6, 8, 12, 16, 20]
bf_p, kmp_p, bmh_p = [], [], []
for pl in pat_lengths:
    p = "".join(random.choices("ACGT", k=pl))
    bf_p.append(brute_force(text_fixed, p).comparisons)
    kmp_p.append(kmp(text_fixed, p).comparisons)
    bmh_p.append(bmh(text_fixed, p).comparisons)

ax3.plot(pat_lengths, bf_p,  "o-", color=COLORS["Brute Force"], linewidth=1.5, markersize=5, label="Brute Force")
ax3.plot(pat_lengths, kmp_p, "s-", color=COLORS["KMP"],         linewidth=1.5, markersize=5, label="KMP")
ax3.plot(pat_lengths, bmh_p, "^-", color=COLORS["BMH"],         linewidth=1.5, markersize=5, label="BMH")
style_ax(ax3, "Comparisons vs pattern length  (text len=500, DNA)", "Pattern length (chars)", "Comparisons")
ax3.legend(fontsize=7)

# ── Chart 4: BF worst-case scaling ──────────────────────────────
wc_sizes = [20, 40, 60, 80, 100, 120, 140]
bf_wc, kmp_wc, bmh_wc = [], [], []
for sz in wc_sizes:
    t = "a" * sz + "b"
    p = "a" * (sz // 5) + "b"
    bf_wc.append(brute_force(t, p).comparisons)
    kmp_wc.append(kmp(t, p).comparisons)
    bmh_wc.append(bmh(t, p).comparisons)

ax4.plot(wc_sizes, bf_wc,  "o-", color=COLORS["Brute Force"], linewidth=1.5, markersize=5, label="Brute Force")
ax4.plot(wc_sizes, kmp_wc, "s-", color=COLORS["KMP"],         linewidth=1.5, markersize=5, label="KMP")
ax4.plot(wc_sizes, bmh_wc, "^-", color=COLORS["BMH"],         linewidth=1.5, markersize=5, label="BMH")
style_ax(ax4, "Worst-case scaling  (text='aaa…b', pattern='aaa…b')", "Text length (chars)", "Comparisons")
ax4.legend(fontsize=7)

# ── Chart 5: per-dataset comparison radar / stacked bar ─────────
ds_labels = ["Short\n(fox)", "Short\n(lazy dog)", "Short\n(brown fox jumps)",
             "Long\n(algorithm)", "Long\n(Boyer-Moore)", "Long\n(preprocessing)",
             "DNA\n(ACGT)", "DNA\n(GATTACA)", "DNA\n(ACGTACGT)"]
x5 = np.arange(len(rows))
w5 = 0.25
ax5.bar(x5 - w5, bf_vals,  w5, color=COLORS["Brute Force"], label="Brute Force", zorder=3)
ax5.bar(x5,      kmp_vals, w5, color=COLORS["KMP"],         label="KMP",         zorder=3)
ax5.bar(x5 + w5, bmh_vals, w5, color=COLORS["BMH"],         label="BMH",         zorder=3)
ax5.set_xticks(x5)
ax5.set_xticklabels(ds_labels, fontsize=7)
style_ax(ax5, "Comparisons per dataset × pattern (detail view)", "Test case", "Comparisons")
ax5.legend(fontsize=7)

# Annotate the winner (lowest bar) in each group
for idx, (b_c, k_c, bm_c) in enumerate(zip(bf_vals, kmp_vals, bmh_vals)):
    mins = {"Brute Force": b_c, "KMP": k_c, "BMH": bm_c}
    winner = min(mins, key=mins.get)
    offsets = {"Brute Force": -w5, "KMP": 0, "BMH": w5}
    ax5.text(idx + offsets[winner], mins[winner] + 0.5, "★",
             ha="center", va="bottom", fontsize=7, color="#2C2C2A")

fig.suptitle("Pattern Search Algorithm Performance Analysis",
             fontsize=14, fontweight="bold", color="#2C2C2A", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])

out_path = "pattern_search_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAF8")
print(f"Chart saved → {out_path}")
