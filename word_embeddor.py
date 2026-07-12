"""
Word2Vec from scratch — Skip-Gram with Negative Sampling (SGNS)
"""

import re
import numpy as np
from collections import Counter

np.random.seed(42)

CORPUS = """
the king is a strong man the queen is a strong woman
the king rules the kingdom the queen rules the kingdom
a man and a woman live in the kingdom
the prince is the son of the king and queen
the princess is the daughter of the king and queen
the boy plays with the dog the girl plays with the cat
the dog barks at the cat the cat runs from the dog
the man walks the dog the woman feeds the cat
paris is the capital of france berlin is the capital of germany
france and germany are countries in europe
the king lives in a big castle the queen lives in a big castle
the prince will become king the princess will become queen
""".strip()


# Tokenize + build vocabulary

def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-z]+", text)


def build_vocab(tokens, min_count: int = 1):
    counts = Counter(tokens)
    # keep words that appear at least `min_count` times
    vocab_words = [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, counts


tokens = tokenize(CORPUS)
word2idx, idx2word, word_counts = build_vocab(tokens, min_count=1)
vocab_size = len(word2idx)


# Generate (target, context) pairs with a sliding window

def generate_pairs(tokens, word2idx, window_size: int = 2):
    ids = [word2idx[t] for t in tokens if t in word2idx]
    pairs = []
    for center_pos, center_id in enumerate(ids):
        window = np.random.randint(1, window_size + 1)  # dynamic window, as in the paper
        start = max(0, center_pos - window)
        end = min(len(ids), center_pos + window + 1)
        for context_pos in range(start, end):
            if context_pos != center_pos:
                pairs.append((center_id, ids[context_pos]))
    return pairs


pairs = generate_pairs(tokens, word2idx, window_size=2)


# Negative sampling distribution (frequency^0.75, as in the paper)

def build_negative_sampling_table(word_counts, word2idx, power: float = 0.75, table_size: int = 100_000):
    freqs = np.zeros(len(word2idx))
    for w, idx in word2idx.items():
        freqs[idx] = word_counts[w]
    freqs = freqs ** power
    freqs /= freqs.sum()
    table = np.random.choice(len(word2idx), size=table_size, p=freqs)
    return table


neg_table = build_negative_sampling_table(word_counts, word2idx)


def sample_negatives(k: int, exclude_id: int):
    negs = []
    while len(negs) < k:
        candidate = neg_table[np.random.randint(len(neg_table))]
        if candidate != exclude_id:
            negs.append(candidate)
    return negs


# Model: two embedding matrices (target, context) + sigmoid loss

EMBEDDING_DIM = 20
LEARNING_RATE = 0.02
EPOCHS = 80
NEG_SAMPLES = 5

W_target = np.random.uniform(-0.5, 0.5, (vocab_size, EMBEDDING_DIM)) / EMBEDDING_DIM
W_context = np.zeros((vocab_size, EMBEDDING_DIM))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train():
    for epoch in range(EPOCHS):
        np.random.shuffle(pairs)
        total_loss = 0.0

        for target_id, context_id in pairs:
            v_t = W_target[target_id]
            v_c = W_context[context_id]

            # positive pair: push sigmoid(v_t . v_c) toward 1
            score = sigmoid(np.dot(v_t, v_c))
            grad = score - 1.0
            grad_t = grad * v_c
            grad_c = grad * v_t
            total_loss += -np.log(score + 1e-10)

            # negative pairs: push sigmoid(v_t . v_n) toward 0
            neg_ids = sample_negatives(NEG_SAMPLES, exclude_id=context_id)
            for neg_id in neg_ids:
                v_n = W_context[neg_id]
                neg_score = sigmoid(np.dot(v_t, v_n))
                grad_t += neg_score * v_n
                W_context[neg_id] -= LEARNING_RATE * neg_score * v_t
                total_loss += -np.log(1 - neg_score + 1e-10)

            W_target[target_id] -= LEARNING_RATE * grad_t
            W_context[context_id] -= LEARNING_RATE * grad_c

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  avg loss = {total_loss / len(pairs):.4f}")


def most_similar(word: str, topn: int = 5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    idx = word2idx[word]
    vec = W_target[idx]
    norms = np.linalg.norm(W_target, axis=1) * np.linalg.norm(vec) + 1e-10
    sims = (W_target @ vec) / norms
    ranked = np.argsort(-sims)
    print(f"\nWords most similar to '{word}':")
    count = 0
    for i in ranked:
        if i == idx:
            continue
        print(f"  {idx2word[i]:12s}  cosine similarity = {sims[i]:.3f}")
        count += 1
        if count >= topn:
            break


if __name__ == "__main__":
    train()

    for w in ["king", "queen", "dog", "france"]:
        most_similar(w, topn=5)