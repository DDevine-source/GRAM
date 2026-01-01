%%writefile baselines_am.py
import numpy as np

def bits01_to_pm1(bits):
    return (bits.astype(np.int8) * 2 - 1)

def hamming_nn(q_bits, bank_bits):
    d = np.sum(np.abs(bank_bits - q_bits), axis=1)
    return int(np.argmin(d)), d

# -------- Modern Hopfield / Attention --------
class ModernHopfieldAttention:
    def __init__(self, bank_bits, beta=10.0):
        self.bank_bits = bank_bits.astype(np.uint8)
        self.bank_pm1  = bits01_to_pm1(self.bank_bits)
        self.P, self.N = self.bank_bits.shape
        self.beta = float(beta)

    def retrieve(self, q_bits):
        q = bits01_to_pm1(q_bits)
        sims = (self.bank_pm1 @ q) / self.N
        z = sims * self.beta
        z -= np.max(z)
        w = np.exp(z); w /= (np.sum(w) + 1e-12)
        y = np.sign((w[:, None] * self.bank_pm1).sum(axis=0))
        y[y == 0] = 1
        y_bits = (y > 0).astype(np.uint8)
        k, _ = hamming_nn(y_bits, self.bank_bits)
        return k

# -------- Dense Associative Memory (toy) --------
class DenseAssociativeMemory:
    def __init__(self, bank_bits, n_order=3, steps=3):
        self.bank_bits = bank_bits.astype(np.uint8)
        self.bank_pm1  = bits01_to_pm1(self.bank_bits)
        self.P, self.N = self.bank_bits.shape
        W = (self.bank_pm1.T @ self.bank_pm1).astype(np.int32)
        np.fill_diagonal(W, 0)
        self.W = W
        self.n = max(3, int(n_order))
        self.steps = max(1, int(steps))

    def retrieve(self, q_bits):
        y = bits01_to_pm1(q_bits).astype(np.int32)
        for _ in range(self.steps):
            Wy = self.W @ y
            g  = np.sign((np.abs(Wy) ** (self.n - 1)) * np.sign(Wy))
            g[g == 0] = 1
            y = g.astype(np.int32)
        y_bits = (y > 0).astype(np.uint8)
        k, _ = hamming_nn(y_bits, self.bank_bits)
        return k

# -------- Gripon–Berrou AM (hardened) --------
class GriponBerrouAM:
    def __init__(self, bank_bits, C=8):
        self.bank_bits = bank_bits.astype(np.uint8)
        self.P, self.N = self.bank_bits.shape
        self.C = int(min(max(1, C), self.N))
        cluster_size = int(np.ceil(self.N / self.C))
        self.starts = list(range(0, self.N, cluster_size))
        self.cluster_indices = []
        for p in range(self.P):
            idxs = []
            bp = self.bank_bits[p]
            for c, s in enumerate(self.starts):
                e = min(self.N, s + cluster_size)
                seg = bp[s:e]
                if seg.size == 0:
                    continue
                if np.any(seg == 1):
                    choice = int(np.argmax(seg))
                else:
                    choice = 0
                pos = s + choice
                if 0 <= pos < self.N:
                    idxs.append((c, pos))
            self.cluster_indices.append(idxs)

    def retrieve(self, q_bits):
        q_use = q_bits if q_bits.size == self.N else np.pad(q_bits, (0, self.N - q_bits.size), mode='constant')[:self.N]
        best_k = 0; best_score = -1
        for k in range(self.P):
            score = 0
            for (_, pos) in self.cluster_indices[k]:
                score += int(q_use[pos] == 1)
            if score > best_score:
                best_score = score; best_k = k
        return best_k

# -------- Willshaw / CMM --------
class WillshawCMM:
    def __init__(self, bank_bits, thresh_frac=0.3):
        self.bank_bits = bank_bits.astype(np.uint8)
        self.P, self.N = self.bank_bits.shape
        M = np.zeros((self.N, self.N), dtype=np.uint8)
        for p in range(self.P):
            x = self.bank_bits[p]
            ix = np.nonzero(x)[0]
            if ix.size:
                M[np.ix_(ix, ix)] = 1
        np.fill_diagonal(M, 0)
        self.M = M
        self.thresh_frac = float(thresh_frac)

    def retrieve(self, q_bits):
        h = self.M @ q_bits
        mx = int(np.max(h)) if h.size else 0
        t = max(1, int(self.thresh_frac * max(1, mx)))
        y_bits = (h >= t).astype(np.uint8)
        k, _ = hamming_nn(y_bits, self.bank_bits)
        return k

# -------- Sparse Distributed Memory (demo) --------
class SparseDistributedMemory:
    def __init__(self, bank_bits, L=256, R=None, seed=0):
        self.bank_bits = bank_bits.astype(np.uint8)
        self.P, self.N = self.bank_bits.shape
        rng = np.random.default_rng(seed)
        self.L = int(L)
        self.addresses = rng.integers(0, 2, size=(self.L, self.N), dtype=np.uint8)
        self.R = int(0.2 * self.N) if R is None else int(R)
        self.contents = np.zeros((self.L, self.N), dtype=np.int32)
        for p in range(self.P):
            x = self.bank_bits[p]
            d = np.sum(np.abs(self.addresses - x), axis=1)
            idx = np.where(d <= self.R)[0]
            if idx.size:
                self.contents[idx] += (x * 2 - 1)

    def retrieve(self, q_bits):
        d = np.sum(np.abs(self.addresses - q_bits), axis=1)
        idx = np.where(d <= self.R)[0]
        y = self.contents[idx].sum(axis=0) if idx.size else self.contents[int(np.argmin(d))]
        y_bits = (y >= 0).astype(np.uint8)
        k, _ = hamming_nn(y_bits, self.bank_bits)
        return k
