"""
A* text alignment plagiarism detector.

Usage: python astar_plagiarism.py
"""

import re
import heapq
import math
from typing import List, Tuple, Dict, Optional

# Text preprocessing utils
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter based on punctuation. Keeps it dependency-free.
    """
    # Normalize newlines and strip
    text = text.strip().replace('\n', ' ')
    # Heuristic: split on sentence-ending punctuation + whitespace
    parts = _SENTENCE_END_RE.split(text)
    # Remove empty and strip
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences

_PUNCT_RE = re.compile(r'[^\w\s]')

def normalize_sentence(sentence: str) -> str:
    """
    Lowercase + remove punctuation (keeps underscores and digits), collapse whitespace.
    """
    s = sentence.lower()
    s = _PUNCT_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(sentence: str) -> List[str]:
    """
    Tokenize by whitespace after normalization.
    """
    s = normalize_sentence(sentence)
    if s == "":
        return []
    return s.split()

# Levenshtein distance (token-level)
def levenshtein_tokens(a: List[str], b: List[str]) -> int:
    """
    Classic DP Levenshtein distance between token lists a and b.
    Returns integer edit distance (insertions/deletions/substitutions).
    """
    na, nb = len(a), len(b)
    if na == 0:
        return nb
    if nb == 0:
        return na
    # Use only two rows for memory efficiency
    prev = list(range(nb + 1))
    cur = [0] * (nb + 1)
    for i in range(1, na + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, nb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,      # deletion from a
                         cur[j - 1] + 1,      # insertion into a
                         prev[j - 1] + cost)  # substitution
        prev, cur = cur, prev
    return prev[nb]

# A* alignment
State = Tuple[int, int]  # (i, j) index positions: next sentence idx in doc1 and doc2

def precompute_distances(doc1_tokens: List[List[str]], doc2_tokens: List[List[str]]) -> List[List[int]]:
    """
    Compute pairwise Levenshtein distances between all sentence tokens of doc1 and doc2.
    """
    n, m = len(doc1_tokens), len(doc2_tokens)
    dist = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dist[i][j] = levenshtein_tokens(doc1_tokens[i], doc2_tokens[j])
    return dist

def heuristic_sum_min_remaining(i: int, j: int, dist: List[List[int]]) -> int:
    """
    Admissible heuristic: sum over remaining sentences of minimal possible match cost.
    For doc1 remaining sentences (i..n-1) we take min over doc2 (j..m-1) and vice versa.
    To remain admissible and symmetric, return max(sum_min1, sum_min2) or average.
    We'll use max(sum_min1, sum_min2) to be safe (still admissible).
    """
    n = len(dist)
    m = len(dist[0]) if dist else 0

    # If empty, heuristic 0
    if n == 0 or m == 0:
        # remaining tokens count triggers minimal costs equal to remaining sentence lengths,
        # but since dist table is empty we just return 0
        return 0

    # sum of min distances for doc1 remaining sentences (over remaining doc2)
    sum_min1 = 0
    for ii in range(i, n):
        # if j...m-1 empty, minimal cost is 0? better to use min over available indices
        if j >= m:
            # no partners left: we could consider deletion cost equal to length of sentence in tokens
            # but we didn't store token lengths here: to keep heuristic admissible and cheap use 0
            # (admissible but weak)
            mincost = 0
        else:
            mincost = min(dist[ii][jj] for jj in range(j, m))
        sum_min1 += mincost

    # sum for doc2 remaining sentences
    sum_min2 = 0
    for jj in range(j, m):
        if i >= n:
            mincost = 0
        else:
            mincost = min(dist[ii][jj] for ii in range(i, n))
        sum_min2 += mincost

    return max(sum_min1, sum_min2)

def a_star_align(doc1_sents: List[str], doc2_sents: List[str]) -> Tuple[int, List[Tuple[Optional[int], Optional[int], int]]]:
    """
    A* alignment between two lists of sentences.
    Returns (total_cost, alignment_path)
    alignment_path is a list of tuples (i_index or None, j_index or None, cost_of_this_action)
    where:
      - (i, j, c) with both i and j not None means align sentence i with j with cost c
      - (i, None, c) means skip sentence i in doc1 (treated as deletion) with cost c
      - (None, j, c) means skip sentence j in doc2 (treated as insertion) with cost c
    Indices in path refer to original sentence indices (0-based).
    """
    # Tokenize
    doc1_tokens = [tokenize(s) for s in doc1_sents]
    doc2_tokens = [tokenize(s) for s in doc2_sents]
    n, m = len(doc1_tokens), len(doc2_tokens)

    # Precompute distances
    if n > 0 and m > 0:
        dist = precompute_distances(doc1_tokens, doc2_tokens)
    else:
        dist = [[0] * m for _ in range(n)]

    # Define cost of skipping a sentence: using token length of sentence as penalty
    # This encourages alignment where possible, but skipping is allowed.
    skip_cost_doc1 = [len(toks) for toks in doc1_tokens]  # deletion cost
    skip_cost_doc2 = [len(toks) for toks in doc2_tokens]  # insertion cost

    # A* frontier: priority queue of (f, g, (i,j), parent_state, action)
    # action encodes how we moved to this state: ('align', i-1,j-1,c), ('skip1', i-1,c), ('skip2', j-1,c)
    start: State = (0, 0)
    # parent map for path reconstruction: maps (i,j) -> (parent_state, action)
    parent: Dict[State, Tuple[Optional[State], Tuple]] = {}

    # best_g so far for a state
    best_g: Dict[State, int] = {start: 0}

    # priority queue entries: (f, g, i, j)
    pq = []
    start_h = heuristic_sum_min_remaining(0, 0, dist) if n and m else 0
    heapq.heappush(pq, (start_h, 0, 0, 0))
    parent[start] = (None, ('start', 0))

    goal_state: Optional[State] = None
    iterations = 0
    max_iters = 1000000  # safety cap

    while pq:
        f, g, i, j = heapq.heappop(pq)
        iterations += 1
        if iterations > max_iters:
            raise RuntimeError("A* exceeded iteration cap")
        state = (i, j)
        # If we popped an outdated entry (worse g than recorded), skip
        if best_g.get(state, math.inf) < g:
            continue

        # Goal check
        if i >= n and j >= m:
            goal_state = state
            break

        # Expand neighbors:
        # 1) Align i with j (if both available)
        if i < n and j < m:
            c = dist[i][j]
            nxt = (i + 1, j + 1)
            g2 = g + c
            if g2 < best_g.get(nxt, math.inf):
                best_g[nxt] = g2
                parent[nxt] = (state, ('align', i, j, c))
                h = heuristic_sum_min_remaining(i + 1, j + 1, dist)
                heapq.heappush(pq, (g2 + h, g2, nxt[0], nxt[1]))

        # 2) Skip sentence in doc1 (delete)
        if i < n:
            c = skip_cost_doc1[i]
            nxt = (i + 1, j)
            g2 = g + c
            if g2 < best_g.get(nxt, math.inf):
                best_g[nxt] = g2
                parent[nxt] = (state, ('skip1', i, c))
                h = heuristic_sum_min_remaining(i + 1, j, dist)
                heapq.heappush(pq, (g2 + h, g2, nxt[0], nxt[1]))

        # 3) Skip sentence in doc2 (insert)
        if j < m:
            c = skip_cost_doc2[j]
            nxt = (i, j + 1)
            g2 = g + c
            if g2 < best_g.get(nxt, math.inf):
                best_g[nxt] = g2
                parent[nxt] = (state, ('skip2', j, c))
                h = heuristic_sum_min_remaining(i, j + 1, dist)
                heapq.heappush(pq, (g2 + h, g2, nxt[0], nxt[1]))

    if goal_state is None:
        # No solution within iterations; return best known?
        # For robustness, pick state with i==n or j==m and reconstruct partial path
        raise RuntimeError("A* failed to find goal (iteration cap or disconnected)")

    # Reconstruct path
    path_actions = []
    cur = goal_state
    while cur != start:
        par, action = parent[cur]
        path_actions.append((cur, action))
        if par is None:
            break
        cur = par
    path_actions.reverse()

    # Convert actions to alignment triplets (i_or_None, j_or_None, cost)
    alignment = []
    for st, action in path_actions:
        kind = action[0]
        if kind == 'align':
            _, i_idx, j_idx, c = action
            alignment.append((i_idx, j_idx, c))
        elif kind == 'skip1':
            _, i_idx, c = action
            alignment.append((i_idx, None, c))
        elif kind == 'skip2':
            _, j_idx, c = action
            alignment.append((None, j_idx, c))
        else:
            # start or unknown
            pass

    total_cost = best_g[goal_state]
    return total_cost, alignment

# Simple plagiarism scoring & pretty printing
def detect_plagiarism(doc1_sents: List[str], doc2_sents: List[str], threshold: int = 3):
    """
    Run alignment and report pairs with edit distance <= threshold as potential plagiarism.
    threshold is in token-level edit distance.
    """
    total_cost, alignment = a_star_align(doc1_sents, doc2_sents)
    print(f"Total alignment cost: {total_cost}")
    print("Aligned pairs (doc1_idx, doc2_idx, cost) [None means skip]:")
    plag_pairs = []
    for tup in alignment:
        i_idx, j_idx, c = tup
        if i_idx is not None and j_idx is not None:
            print(f"  Doc1[{i_idx}] <-> Doc2[{j_idx}]  cost={c}")
            if c <= threshold:
                plag_pairs.append((i_idx, j_idx, c))
        elif i_idx is not None:
            print(f"  Doc1[{i_idx}]  -- skipped --  cost={c}")
        else:
            print(f"  Doc2[{j_idx}]  -- skipped --  cost={c}")
    if plag_pairs:
        print("\nPotential plagiarism detected for the following sentence pairs (cost <= {}):".format(threshold))
        for i,j,c in plag_pairs:
            print(f"  Doc1[{i}]: {doc1_sents[i]}\n  Doc2[{j}]: {doc2_sents[j]}\n  edit_tokens={c}\n")
    else:
        print("\nNo low-cost alignments detected (threshold {}).".format(threshold))

# --------------------------
# Test cases
# --------------------------
def run_test_cases():
    # Case 1: Slightly Modified
    docA1 = "Two roads diverged in a yellow wood. And sorry I could not travel both. And be one traveler, long I stood."
    docB1 = "Two paths diverged in a yellow wood. And sorry I could not travel the two. And as one traveler, I stood for a long time."
    sentsA1 = split_sentences(docA1)
    sentsB1 = split_sentences(docB1)
    print("\n=== Test Case 1: Slightly Modified ===")
    detect_plagiarism(sentsA1, sentsB1, threshold=4)

    # Case 2: Reordered and Modified
    docA2 = "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character. I have a dream today! This is our hope."
    docB2 = "This is our great hope. I have a dream today! I have a powerful dream that my four small children will one day live in a country where they are not judged by their skin color but by the quality of their character."
    sentsA2 = split_sentences(docA2)
    sentsB2 = split_sentences(docB2)
    print("\n=== Test Case 2: Reordered and Modified ===")
    detect_plagiarism(sentsA2, sentsB2, threshold=6)

    # Case 3: Completely Different
    docA3 = "I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion. All those moments will be lost in time, like tears in rain."
    docB3 = "May the Force be with you. It's a trap! I am your father."
    sentsA3 = split_sentences(docA3)
    sentsB3 = split_sentences(docB3)
    print("\n=== Test Case 3: Completely Different ===")
    detect_plagiarism(sentsA3, sentsB3, threshold=5)

    # Case 4: Partial Overlap
    docA4 = "O Captain! my Captain! our fearful trip is done. The ship has weather'd every rack, the prize we sought is won. The port is near, the bells I hear, the people all exulting."
    docB4 = "Here's looking at you, kid. The prize we sought is won. Frankly, my dear, I don't give a damn."
    sentsA4 = split_sentences(docA4)
    sentsB4 = split_sentences(docB4)
    print("\n=== Test Case 4: Partial Overlap - Mixed===")
    detect_plagiarism(sentsA4, sentsB4, threshold=3)

if __name__ == "__main__":
    run_test_cases()
