import numpy as np
import pickle
import os
import random
import csv
from itertools import combinations
import galois

# ---------------- Helper functions ----------------

# Function: genSMatrix - Helper routine for genSMatrix logic.
# Parameters: `k` is message dimension or generic rank parameter.
def genSMatrix(k):
    """Generates an invertible matrix k x k (over integers 0/1)"""
    sMaybe = np.matrix(np.random.randint(0, 2, k * k).reshape(k, k).astype(int))
    while True:
        try:
            sMaybe.getI()
            return sMaybe
        except Exception:
            sMaybe = np.matrix(np.random.randint(0, 2, k * k).reshape(k, k).astype(int))

# Function: genPMatrix - Helper routine for genPMatrix logic.
# Parameters: `n` is code length or generic size parameter; `keep` is keep input value.
def genPMatrix(n, keep=False):
    """Generates a permutation matrix n x n"""
    p = np.identity(n, dtype=int)
    if keep:
        return np.matrix(p).reshape(n, n)
    else:
        perm = np.random.permutation(p)
        return np.matrix(perm)

# Function: modTwo - Helper routine for modTwo logic.
# Parameters: `C` is C input value.
def modTwo(C):
    """Return C mod 2 """
    D = np.array(C).copy()
    D.fill(2)
    return np.remainder(np.array(C, dtype=int), 2)

# Function: bitFlip - Helper routine for bitFlip logic.
# Parameters: `C` is C input value; `n` is code length or generic size parameter.
def bitFlip(C, n):
    """
    Flips a bit in C. C expected shape (1,n) as numpy.matrix or numpy.array.
    n: 0 -> no flip, -1 -> random flip, positive -> 1-indexed position to flip
    Returns same type as input (matrix).
    """
    if n == 0:
        return C
    if isinstance(C, np.matrix):
        arr = np.array(C, dtype=int)
        is_matrix = True
    else:
        arr = np.array(C, dtype=int)
        is_matrix = False

    if n == -1:
        idx = random.randint(0, arr.shape[1] - 1)
    else:
        idx = int(n) - 1

    arr[0, idx] ^= 1

    return np.matrix(arr) if is_matrix else arr

# Function: all_zeros - Helper routine for all zeros logic.
# Parameters: `d` is d input value.
def all_zeros(d):
    """Return True if all elements in iterable d are zero"""
    return all(int(x) == 0 for x in np.ravel(d))

# Function: checkOldGuesses - Helper routine for checkOldGuesses logic.
# Parameters: `oG` is oG input value; `newGuess` is newGuess input value.
def checkOldGuesses(oG, newGuess):
    """Return False if newGuess.A1 matches any in oG; True otherwise"""
    for s in oG:
        if np.array_equal(newGuess.A1, s.A1):
            return False
    return True

# Function: makeString - Helper routine for makeString logic.
# Parameters: `matrix` is matrix input value.
def makeString(matrix):
    """Make a simple string representation of numpy matrix (flattened bits)"""
    #return "".join(str(int(np.int64(x))) for x in np.array(matrix).ravel().tolist())
    return "".join(map(str, np.array(matrix, dtype=int).ravel().tolist()))


# Function: generate_error_vector_weight - Helper routine for generate error vector weight logic.
# Parameters: `n` is code length or generic size parameter; `weight` is weight input value.
def generate_error_vector_weight(n, weight):
    """Generate error vector of specified weight as numpy.matrix(1 x n)"""
    weight = max(0, min(weight, n))
    positions = random.sample(range(n), weight) if weight > 0 else []
    e = np.zeros((1, n), dtype=int)
    for p in positions:
        e[0, p] = 1
    return np.matrix(e)

# ---------------- Goppa demo implementation ----------------

class GoppaDecoder:
    """
    Demo Goppa decoder:
      - Builds GF(2^m), selects a small g(x) (demo), builds support L
      - Builds parity-check H over GF(2^m) and expands to binary H_binary (t*m x n)
      - Builds a binary generator matrix G via null-space of H (small sizes)
      - decode_syndrome: brute-force search up to weight t (small n)
    Limitations: brute-force decode; no Patterson; only demo sizes feasible.
    """

    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `m` is field/message parameter used by the algorithm; `t` is error-correction strength or threshold parameter.
    def __init__(self, m=4, t=2):
        self.m = int(m)
        self.t = int(t)
        self.GF = galois.GF(2**self.m)

        # pick Goppa polynomial (simple deterministic small choices)
        self.g = self._generate_goppa_polynomial()

        # build support L (exclude elements that are roots of g)
        self.L = self._generate_support()
        self.n = len(self.L)

        # build H in GF and expanded binary H
        self.H_field = self._build_parity_check_over_gf()     # shape (t, n) over GF(2^m)
        self.H_binary = self._expand_to_binary(self.H_field)  # shape (t*m, n), dtype uint8

        # derive binary generator matrix G (small demo only)
        self.G_binary, self.k = self.compute_generator_matrix(np.matrix(self.H_binary))

    # Method: _generate_goppa_polynomial - Helper routine for  generate goppa polynomial logic.
    # Parameters: `self` is class instance reference.
    def _generate_goppa_polynomial(self):
        # small fixed examples for demo
        # degree t required
        if self.m == 4 and self.t == 2:
            # x^2 + x + 1
            coeffs = [1, 1, 1]
        elif self.m == 4 and self.t == 3:
            coeffs = [1, 1, 0, 1]
        else:
            # fallback: choose polynomial x^t + x + 1 (may not be irreducible)
            coeffs = [1] + [0] * (self.t - 2) + [1, 1] if self.t >= 2 else [1, 1]
        return galois.Poly(coeffs, field=self.GF)

    # Method: _generate_support - Helper routine for  generate support logic.
    # Parameters: `self` is class instance reference.
    def _generate_support(self):
        support = []
        for a in self.GF.elements:
            if self.g(a) != 0:
                support.append(a)
        # trim if too long (keep at most 2^m - 1)
        return support[: (2**self.m) - 1]

    # Method: _build_parity_check_over_gf - Helper routine for  build parity check over gf logic.
    # Parameters: `self` is class instance reference.
    def _build_parity_check_over_gf(self):
        n = len(self.L)
        H = np.zeros((self.t, n), dtype=self.GF)
        for i, alpha in enumerate(self.L):
            denom = self.g(alpha)
            if denom == 0:
                raise ValueError("Support element is a root of g(x); choose different support or g.")
            inv = self.GF(1) / denom
            pow_alpha = self.GF(1)
            for j in range(self.t):
                H[j, i] = pow_alpha * inv
                pow_alpha *= alpha
        return H

    # Method: _expand_to_binary - Helper routine for  expand to binary logic.
    # Parameters: `self` is class instance reference; `H_gf` is H gf input value.
    def _expand_to_binary(self, H_gf):
        t, n = H_gf.shape
        H_bin = np.zeros((t * self.m, n), dtype=np.uint8)
        for j in range(t):
            for i in range(n):
                val = int(np.int64(H_gf[j, i]))  # integer representation
                bits = [(val >> b) & 1 for b in range(self.m)]  # LSB-first
                H_bin[j*self.m:(j+1)*self.m, i] = bits
        return H_bin

    # Method: compute_parity_check_matrix - Helper routine for compute parity check matrix logic.
    # Parameters: `self` is class instance reference.
    def compute_parity_check_matrix(self):
        """Return binary parity-check matrix as numpy.matrix (t*m x n)"""
        return np.matrix(self.H_binary.astype(int))

    # Method: compute_generator_matrix - Helper routine for compute generator matrix logic.
    # Parameters: `self` is class instance reference; `H` is H input value.
    def compute_generator_matrix(self, H):
        """
        Compute generator matrix G over GF(2) as null-space of H (binary).
        Returns (G_matrix, k).
        For small matrices only; uses simple GF(2) linear algebra.
        """
        H_arr = np.array(H, dtype=int) & 1
        N = self.gf2_null_space(H_arr)
        if N.size == 0:
            return np.matrix(np.zeros((0, H_arr.shape[1]), dtype=int)), 0
        G = np.matrix(N.astype(int))
        k = G.shape[0]
        return G, k

    # ----- gf2 linear algebra helpers -----
    # Method: _gf2_rref - Helper routine for  gf2 rref logic.
    # Parameters: `self` is class instance reference; `A` is A input value.
    def _gf2_rref(self, A):
        A = (A.copy() & 1).astype(np.uint8)
        m, n = A.shape
        r = 0
        pivots = []
        for c in range(n):
            pivot = None
            for rr in range(r, m):
                if A[rr, c]:
                    pivot = rr
                    break
            if pivot is None:
                continue
            if pivot != r:
                A[[r, pivot]] = A[[pivot, r]]
            pivots.append(c)
            for rr in range(m):
                if rr != r and A[rr, c]:
                    A[rr, :] ^= A[r, :]
            r += 1
            if r == m:
                break
        return A, pivots

    # Method: gf2_null_space - Helper routine for gf2 null space logic.
    # Parameters: `self` is class instance reference; `A` is A input value.
    def gf2_null_space(self, A):
        A = (A & 1).astype(np.uint8)
        m, n = A.shape
        R, pivots = self._gf2_rref(A.copy())
        pivot_set = set(pivots)
        free_cols = [c for c in range(n) if c not in pivot_set]
        if not free_cols:
            return np.zeros((0, n), dtype=np.uint8)
        basis = []
        for f in free_cols:
            v = np.zeros(n, dtype=np.uint8)
            v[f] = 1
            # back-substitute pivot variables
            for r in range(R.shape[0] - 1, -1, -1):
                row = R[r]
                ones = np.flatnonzero(row)
                if ones.size == 0:
                    continue
                pivot_col = ones[0]
                s = 0
                for j in range(pivot_col + 1, n):
                    if row[j] and v[j]:
                        s ^= 1
                v[pivot_col] = s
            basis.append(v)
        return np.vstack(basis).astype(np.uint8)

    # ----- demo decoder (brute-force up to weight t) -----
    # Method: decode_syndrome - Helper routine for decode syndrome logic.
    # Parameters: `self` is class instance reference; `syndrome_matrix` is syndrome matrix input value.
    def decode_syndrome(self, syndrome_matrix):
        """
        Brute-force decode: try error patterns up to weight t (limited).
        syndrome_matrix: numpy.matrix shape (t*m, 1) or 1D array length t*m
        Returns numpy.matrix(1 x n) error vector or None.
        """
        if isinstance(syndrome_matrix, np.matrix):
            syndrome = np.array(syndrome_matrix).ravel() & 1
        else:
            syndrome = np.array(syndrome_matrix).ravel() & 1
        if np.all(syndrome == 0):
            return None

        H_bin = self.H_binary  # shape (t*m, n)
        n = self.n

        # brute-force up to weight t but cap at small maximum to avoid explosion
        max_search_weight = min(self.t, 4)
        for w in range(1, max_search_weight + 1):
            for pos in combinations(range(n), w):
                e = np.zeros(n, dtype=int)
                e[list(pos)] = 1
                syn = (H_bin @ e) & 1
                if np.array_equal(syn, syndrome.astype(np.uint8)):
                    return np.matrix(e.reshape(1, -1), dtype=int)
        return None

# ---------------- syndrome lookup wrapper ----------------

# Function: syndromeLookupGoppa - Helper routine for syndromeLookupGoppa logic.
# Parameters: `H_bin_matrix` is H bin matrix input value; `cHat` is cHat input value; `goppa_decoder` is goppa decoder input value.
def syndromeLookupGoppa(H_bin_matrix, cHat, goppa_decoder):
    """
    H_bin_matrix: numpy.matrix (t*m x n) binary
    cHat: numpy.matrix shape (1,n)
    goppa_decoder: instance of GoppaDecoder
    Returns: 0 if no error, else 1-indexed first error position (like original)
    """
    # compute binary syndrome
    syndrome = modTwo(np.matrix(H_bin_matrix) * np.matrix(cHat).T)  # shape (t*m, 1)
    if all_zeros(np.ravel(syndrome)):
        return 0
    err = goppa_decoder.decode_syndrome(syndrome)
    if err is None:
        return 0
    arr = np.array(err).ravel()
    ones = np.flatnonzero(arr)
    if ones.size == 0:
        return 0
    
    return int(ones[0]) + 1  # 1-indexed to match original bitFlip behavior

# ---------------- Private/Public key classes (Goppa) ----------------

class privateKeyH84:
    """
    Re-uses name for compatibility. This is now a Goppa-based private key (demo).
    Interfaces (makeGPrime, decrypt, decryptFile, dnaFileDecrypt) match original.
    """
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `S` is S input value; `P` is P input value; `m` is field/message parameter used by the algorithm; `t` is error-correction strength or threshold parameter.
    def __init__(self, S=None, P=None, m=4, t=2):
        self.goppa_decoder = GoppaDecoder(m=m, t=t)
        self.m = int(m)
        self.t = int(t)
        self.n = self.goppa_decoder.n
        # binary matrices
        self.H = self.goppa_decoder.compute_parity_check_matrix()
        self.G = self.goppa_decoder.G_binary
        self.k = self.goppa_decoder.k

        if S is None:
            self.S = modTwo(genSMatrix(self.k))
        else:
            self.S = S

        if P is None:
            self.P = modTwo(genPMatrix(self.n))
        else:
            self.P = P

    # Method: printCode - Helper routine for printCode logic.
    # Parameters: `self` is class instance reference.
    def printCode(self):
        print("S:\n", self.S)
        print("P:\n", self.P)
        print("GPrime:\n", self.makeGPrime())
        print(f"Goppa params: m={self.m}, t={self.t}, n={self.n}, k={self.k}")

    # Method: writeKeyToFile - Helper routine for writeKeyToFile logic.
    # Parameters: `self` is class instance reference; `keyFile` is keyFile input value.
    def writeKeyToFile(self, keyFile):
        try:
            pickle.dump(self, open(keyFile, "wb"))
        except Exception as e:
            print("Could not save key file to:", keyFile, e)
            raise

    # Method: readKeyFromFile - Helper routine for readKeyFromFile logic.
    # Parameters: `self` is class instance reference; `keyFile` is keyFile input value.
    def readKeyFromFile(self, keyFile):
        try:
            newPriv = pickle.load(open(keyFile, "rb"))
            self.S = newPriv.S
            self.P = newPriv.P
            self.G = newPriv.G
            self.H = newPriv.H
            self.goppa_decoder = newPriv.goppa_decoder
            self.m = newPriv.m
            self.t = newPriv.t
            self.n = newPriv.n
            self.k = newPriv.k
        except Exception as e:
            print("Could not load key file:", keyFile, e)
            raise

    # Method: makeGPrime - Helper routine for makeGPrime logic.
    # Parameters: `self` is class instance reference.
    def makeGPrime(self):
        return modTwo(self.S * self.G * self.P)

    # Method: decrypt - Helper routine for decrypt logic.
    # Parameters: `self` is class instance reference; `c` is c input value.
    def decrypt(self, c):
        """
        c: numpy.matrix (1, n)
        returns: numpy.matrix (1, k) plaintext bits
        """
        # undo permutation
        cHat = c * modTwo(np.linalg.inv(np.array(self.P, dtype=int)) % 2)
        
        # use goppa decoder to find first error position
        error_pos = syndromeLookupGoppa(self.H, cHat, self.goppa_decoder)
        
		# correct
        m_mat = bitFlip(cHat, error_pos)
        
		# extract message (first k bits) and undo S
        m_bits = modTwo(m_mat[0, 0:self.k] * (np.linalg.inv(np.array(self.S, dtype=int)) % 2))

        return m_bits

    # Method: decryptFile - Helper routine for decryptFile logic.
    # Parameters: `self` is class instance reference; `f` is file path or file-like input.
    def decryptFile(self, f):
        with open(f, "rb") as cf, open(f + ".decoded", "wb") as mf:
            cb1 = cf.read(1)
            cb2 = cf.read(1)
            while cb1 and cb2:
                b1 = cb1[0]
                bits1 = '{0:08b}'.format(b1)
                c1_l = [int(s) for s in bits1]
                c1_m = np.matrix(c1_l, dtype=int)
                # pad/truncate
                if c1_m.shape[1] < self.n:
                    pad = np.zeros((1, self.n - c1_m.shape[1]), dtype=int)
                    c1_m = np.hstack([c1_m, pad])
                elif c1_m.shape[1] > self.n:
                    c1_m = c1_m[:, :self.n]
                d1 = self.decrypt(c1_m)
                m1 = "".join(str(int(d1.item(i))) for i in range(min(d1.size, self.k)))

                b2 = cb2[0]
                bits2 = '{0:08b}'.format(b2)
                c2_l = [int(s) for s in bits2]
                c2_m = np.matrix(c2_l, dtype=int)
                if c2_m.shape[1] < self.n:
                    pad = np.zeros((1, self.n - c2_m.shape[1]), dtype=int)
                    c2_m = np.hstack([c2_m, pad])
                elif c2_m.shape[1] > self.n:
                    c2_m = c2_m[:, :self.n]
                d2 = self.decrypt(c2_m)
                m2 = "".join(str(int(d2.item(i))) for i in range(min(d2.size, self.k)))

                combined = m1 + m2
                if len(combined) >= 8:
                    mf.write(bytes([int(combined[:8], 2)]))

                cb1 = cf.read(1)
                cb2 = cf.read(1)

    # Method: dnaFileDecrypt - Helper routine for dnaFileDecrypt logic.
    # Parameters: `self` is class instance reference; `f` is file path or file-like input; `dlu` is dlu input value.
    def dnaFileDecrypt(self, f, dlu):
        with open(f, "r") as cf, open(f + ".decoded", "w") as mf:
            c1 = cf.readline().strip("\n")
            c2 = cf.readline().strip("\n")
            while c1 and c2:
                mat1 = np.matrix(" ".join(dlu.lookDNADecrypt(c1)), dtype=int)
                mat2 = np.matrix(" ".join(dlu.lookDNADecrypt(c2)), dtype=int)
                # pad/truncate
                if mat1.shape[1] < self.n:
                    pad = np.zeros((1, self.n - mat1.shape[1]), dtype=int)
                    mat1 = np.hstack([mat1, pad])
                elif mat1.shape[1] > self.n:
                    mat1 = mat1[:, :self.n]
                if mat2.shape[1] < self.n:
                    pad = np.zeros((1, self.n - mat2.shape[1]), dtype=int)
                    mat2 = np.hstack([mat2, pad])
                elif mat2.shape[1] > self.n:
                    mat2 = mat2[:, :self.n]

                d1 = self.decrypt(mat1)
                d2 = self.decrypt(mat2)
                m1 = "".join(str(int(d1.item(i))) for i in range(min(d1.size, self.k)))
                m2 = "".join(str(int(d2.item(i))) for i in range(min(d2.size, self.k)))
                combined = m1 + m2
                if len(combined) >= 8:
                    mf.write(chr(int(combined[:8], 2)))
                c1 = cf.readline().strip("\n")
                c2 = cf.readline().strip("\n")

# ---------------- Goppa Public Key ----------------

class publicKeyH84:
    """Public key (Goppa demo) with same public methods as original"""
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `GPrime` is GPrime input value; `t` is error-correction strength or threshold parameter.
    def __init__(self, GPrime, t=2):
        self.GPrime = GPrime
        self.t = t
        self.error_vector_weight = t
        print(f"Goppa (demo) error correction capability: {self.error_vector_weight}")

    # Method: printCode - Helper routine for printCode logic.
    # Parameters: `self` is class instance reference.
    def printCode(self):
        print("GPrime:\n", self.GPrime)

    # Method: writeKeyToFile - Helper routine for writeKeyToFile logic.
    # Parameters: `self` is class instance reference; `keyFile` is keyFile input value.
    def writeKeyToFile(self, keyFile):
        try:
            pickle.dump(self, open(keyFile, "wb"))
        except Exception as e:
            print("Could not save key file to:", keyFile, e)
            raise

    # Method: readKeyFromFile - Helper routine for readKeyFromFile logic.
    # Parameters: `self` is class instance reference; `keyFile` is keyFile input value.
    def readKeyFromFile(self, keyFile):
        try:
            newPub = pickle.load(open(keyFile, "rb"))
            self.GPrime = newPub.GPrime
            self.t = getattr(newPub, "t", 2)
            self.error_vector_weight = getattr(newPub, "error_vector_weight", self.t)
        except Exception as e:
            print("Could not load key file:", keyFile, e)
            raise

    # Method: encrypt - Helper routine for encrypt logic.
    # Parameters: `self` is class instance reference; `m` is field/message parameter used by the algorithm.
    def encrypt(self, m):
        """m: numpy.matrix (1, k)"""
        c_encoded = modTwo(m * self.GPrime)
        error_vec = generate_error_vector_weight(int(c_encoded.shape[1]), self.error_vector_weight)
        c = modTwo(c_encoded + error_vec)
        return c

    # Method: encryptFile - Helper routine for encryptFile logic.
    # Parameters: `self` is class instance reference; `f` is file path or file-like input.
    def encryptFile(self, f):
        with open(f, "rb") as mf, open(f + ".ctxt", "wb") as cf:
            b = mf.read(1)
            while b:
                byte = b[0]
                bits = '{0:08b}'.format(byte)
                k = int(self.GPrime.shape[0])
                for i in range(0, len(bits), k):
                    chunk = bits[i:i+k]
                    while len(chunk) < k:
                        chunk += '0'
                    m_bits = [int(ch) for ch in chunk[:k]]
                    m_mat = np.matrix(m_bits, dtype=int)
                    c_mat = self.encrypt(m_mat)
                    c_bits = "".join(str(int(c_mat.item(d))) for d in range(int(c_mat.size)))
                    for j in range(0, len(c_bits), 8):
                        byte_chunk = c_bits[j:j+8]
                        if len(byte_chunk) == 8:
                            cf.write(bytes([int(byte_chunk, 2)]))
                b = mf.read(1)

# ---------------- Keep original Hamming classes for compatibility ----------------

class privateKeyH1611:
    """Hamming 16,11 """
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `S` is S input value; `P` is P input value.
    def __init__(self, S=None, P=None):
        self.G = np.matrix([
            [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
            [0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1],
            [0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1],
            [0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1],
            [0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1],
            [0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1],
            [0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1],
            [0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0],
            [0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0]
        ], dtype=int)

        self.H = np.matrix([
            [1,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1],
            [0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1],
            [0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1],
            [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0]
        ], dtype=int)

        if S is None:
            self.S = modTwo(genSMatrix(11))
        else:
            self.S = S
        if P is None:
            self.P = modTwo(genPMatrix(16))
        else:
            self.P = P

    # Method: printCode - Helper routine for printCode logic.
    # Parameters: `self` is class instance reference.
    def printCode(self):
        print("S:\n", self.S)
        print("P:\n", self.P)
        print("GPrime:\n", self.makeGPrime())

    # Method: makeGPrime - Helper routine for makeGPrime logic.
    # Parameters: `self` is class instance reference.
    def makeGPrime(self):
        return modTwo(self.S * self.G * self.P)

class publicKeyH1611:
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `GPrime` is GPrime input value.
    def __init__(self, GPrime):
        self.GPrime = GPrime
    def printCode(self):
        print("GPrime:\n", self.GPrime)
    # Method: encrypt - Helper routine for encrypt logic.
    # Parameters: `self` is class instance reference; `m` is field/message parameter used by the algorithm.
    def encrypt(self, m):
        z = random.randint(1, 15)
        c = bitFlip(modTwo(m * self.GPrime), z)
        return c

# ---------------- Brute forcer (keeps interface, demo only) ----------------

class bruteForcerH84:
    """Tries to brute-force S,P given GPrime (demo only)"""
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `GPrime` is GPrime input value.
    def __init__(self, GPrime):
        self.attempts = 0
        self.GPrime = GPrime
        self.GPrimeConsider = 0
        self.sConsider = 0
        self.pConsider = 0
        self.STries = list()
        self.PTries = list()
        # prepare a default small Goppa code to attempt against
        decoder = GoppaDecoder(m=4, t=2)
        self.G, k = decoder.compute_generator_matrix(decoder.compute_parity_check_matrix())

    # Method: attemptKey - Helper routine for attemptKey logic.
    # Parameters: `self` is class instance reference.
    def attemptKey(self):
        self.attempts = 1
        k = self.G.shape[0]
        n = self.G.shape[1]
        self.sConsider = genSMatrix(k)
        self.STries.append(self.sConsider)
        self.pConsider = genPMatrix(n)
        self.PTries.append(self.pConsider)
        self.GPrimeConsider = modTwo(self.sConsider * self.G * self.pConsider)
        while not np.array_equal(np.ravel(self.GPrimeConsider), np.ravel(self.GPrime)):
            self.attempts += 1
            self.sConsider = genSMatrix(k)
            while not checkOldGuesses(self.STries, self.sConsider):
                self.sConsider = genSMatrix(k)
            self.STries.append(self.sConsider)
            self.pConsider = genPMatrix(n)
            while not checkOldGuesses(self.PTries, self.pConsider):
                self.pConsider = genPMatrix(n)
            self.PTries.append(self.pConsider)
            self.GPrimeConsider = modTwo(self.sConsider * self.G * self.pConsider)
        return True

# ---------------- DNA lookup ----------------

class lookupDNA:
    """Data structure for DNA bit mapping (unchanged)"""
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `encryptFile` is encryptFile input value; `decryptFile` is decryptFile input value.
    def __init__(self, encryptFile, decryptFile):
        self.encLU = dict()
        self.decLU = dict()
        dreader = csv.reader(open(decryptFile, 'r'))
        for row in dreader:
            k, v = row
            self.decLU[k] = str(v)
        ereader = csv.reader(open(encryptFile, 'r'))
        for row in ereader:
            k, v = row
            self.encLU[k] = str(v)
    # Method: lookDNAEncrypt - Helper routine for lookDNAEncrypt logic.
    # Parameters: `self` is class instance reference; `bstring` is bstring input value.
    def lookDNAEncrypt(self, bstring):
        return str(self.encLU.get(bstring, "?"))
    def lookDNADecrypt(self, bstring):
        return str(self.decLU.get(bstring, "?"))

# ---------------- Quick main test (demo) ----------------

if __name__ == "__main__":
    # pass

    priv = privateKeyH84(m=4, t=2)
    pub = publicKeyH84(priv.makeGPrime(), t=2)
    print(f"params: m={priv.m}, t={priv.t}, n={priv.n}, k={priv.k}")
    # prepare message (k bits)
    msg_bits = [1, 0, 1, 0] + [0] * max(0, priv.k - 4)
    msg = np.matrix(msg_bits[:priv.k], dtype=int)
    print("message:", msg)
    ct = pub.encrypt(msg)
    print("ciphertext:", ct)
    pt = priv.decrypt(ct)
    print("decrypted (bits):", pt)
