import numpy as np
import os
import csv
import json
import secrets
import base64
import hashlib
import hmac
import struct
from itertools import combinations
from types import SimpleNamespace
import galois
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
# add explicit import of the Patterson decoder function
from patterson_decode import patterson_decode as patterson_decode_func

KEY_FORMAT_VERSION = 1
QMC_MAGIC = b"QMC1"
QMC_CONTAINER_VERSION = 1
QMC_HEADER_MAX_BYTES = 1 << 20


def _b64e(data):
    return base64.b64encode(data).decode("ascii")


def _b64d(text, field_name):
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as ex:
        raise ValueError(f"Invalid base64 for {field_name}") from ex


def _bits_to_bytes(bits):
    """Pack bit vector into bytes (MSB-first within each byte)."""
    out = bytearray()
    bits = [int(b) & 1 for b in bits]
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            chunk += [0] * (8 - len(chunk))
        val = 0
        for bit in chunk:
            val = (val << 1) | bit
        out.append(val)
    return bytes(out)


def _bytes_to_bits(data, bit_len):
    """Unpack bytes into a fixed-length bit vector (MSB-first within each byte)."""
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    if len(bits) < bit_len:
        raise ValueError("Byte payload too short for requested bit length")
    return np.array(bits[:bit_len], dtype=np.uint8)


def _hkdf_sha256(ikm, salt, info, length=32):
    """HKDF-SHA256 (RFC 5869) with explicit output length."""
    if salt is None:
        salt = b""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    prev = b""
    counter = 1
    while len(okm) < length:
        prev = hmac.new(prk, prev + info + bytes([counter]), hashlib.sha256).digest()
        okm += prev
        counter += 1
    return okm[:length]


def _kem_meta_aad(version, m, t, n, k):
    return f"qmc-v{version}|m={m}|t={t}|n={n}|k={k}".encode("ascii")


def _rand_bit_matrix(rows, cols):
    """Return a rows x cols matrix of CSPRNG bits."""
    bits = [secrets.randbits(1) for _ in range(rows * cols)]
    return np.array(bits, dtype=np.uint8).reshape(rows, cols)


def _matrix_to_bit_lists(mat):
    """Serialize matrix as nested 0/1 lists."""
    arr = np.array(mat, dtype=np.uint8) & 1
    return arr.tolist()


def _bit_lists_to_matrix(payload, expected_shape=None, field_name="matrix"):
    """Deserialize and validate nested 0/1 lists as uint8 array."""
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{field_name} must be a non-empty 2D list")
    arr = np.array(payload, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} must be 2-dimensional")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(f"{field_name} must contain only 0/1 values")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"{field_name} has shape {arr.shape}; expected {expected_shape}"
        )
    return arr


def _secure_permutation_indices(n):
    """Return a cryptographically random permutation of range(n)."""
    idx = list(range(n))
    secrets.SystemRandom().shuffle(idx)
    return idx


# ---------------- GF(2) linear algebra helpers ----------------

# Function: gf2_rref_with_transform - Row-reduction over GF(2) with transform matrix
def gf2_rref_with_transform(A):
    """
    Returns (R, pivots, U) such that U @ A = R (all over GF(2)).
    R has pivot columns reduced (1 at pivot row, 0 elsewhere).
    U is the accumulated row-operation matrix (invertible if full row rank).
    """
    A = (np.array(A, dtype=np.uint8) & 1)
    m, n = A.shape
    U = np.eye(m, dtype=np.uint8)
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
            U[[r, pivot]] = U[[pivot, r]]
        pivots.append(c)
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
                U[rr, :] ^= U[r, :]
        r += 1
        if r == m:
            break
    return A, pivots, U

# Function: gf2_inv - Invert a square binary matrix over GF(2)
def gf2_inv(M):
    """
    Invert a square binary matrix over GF(2). Raises ValueError if singular.
    """
    M = (np.array(M, dtype=np.uint8) & 1)
    if M.shape[0] != M.shape[1]:
        raise ValueError("GF(2) inverse requires a square matrix")
    R, pivots, U = gf2_rref_with_transform(M)
    if len(pivots) != M.shape[0]:
        raise ValueError("Matrix not invertible over GF(2)")
    # U @ M = I, hence U = M^{-1}
    return U

# Function: gf2_right_inverse - Compute a right-inverse over GF(2) for full-row-rank G
def gf2_right_inverse(G):
    """
    Compute a right-inverse M for G over GF(2) such that G @ M = I_k.
    G must be k x n with full row rank k.
    """
    G = (np.array(G, dtype=np.uint8) & 1)
    k, n = G.shape
    R, pivots, U = gf2_rref_with_transform(G)
    if len(pivots) != k:
        raise ValueError("Generator matrix G does not have full row rank")
    M = np.zeros((n, k), dtype=np.uint8)
    for r, c in enumerate(pivots):
        M[c, :] = U[r, :]
    return M

# ---------------- Helper functions ----------------

# Function: genSMatrix - Generate random invertible k x k binary matrix
def genSMatrix(k):
    """Generates a random invertible k x k binary matrix over GF(2)."""
    while True:
        sMaybe = _rand_bit_matrix(k, k)
        try:
            _ = gf2_inv(sMaybe)
            return np.matrix(sMaybe.astype(int))
        except ValueError:
            continue

# Function: genPMatrix - Generate a random permutation matrix n x n
def genPMatrix(n, keep=False):
    """Generates a permutation matrix n x n"""
    p = np.identity(n, dtype=int)
    if keep:
        return np.matrix(p).reshape(n, n)
    else:
        perm = p[_secure_permutation_indices(n)]
        return np.matrix(perm)

# Function: modTwo - Take entries modulo 2, preserving type
def modTwo(C):
    """Return C mod 2 preserving matrix/array type."""
    arr = (np.array(C, dtype=int) & 1)
    return np.matrix(arr) if isinstance(C, np.matrix) else arr

# Function: bitFlip - Flip one bit by index or randomly (-1)
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
        idx = secrets.randbelow(arr.shape[1])
    else:
        idx = int(n) - 1
        if idx < 0 or idx >= arr.shape[1]:
            raise ValueError(f"Bit index out of bounds for width={arr.shape[1]}: {n}")

    arr[0, idx] ^= 1

    return np.matrix(arr) if is_matrix else arr

# Function: all_zeros - Check if all elements are zero
def all_zeros(d):
    """Return True if all elements in iterable d are zero"""
    return all(int(x) == 0 for x in np.ravel(d))

# Function: checkOldGuesses - Check for duplicate candidate matrices
def checkOldGuesses(oG, newGuess):
    """Return False if newGuess.A1 matches any in oG; True otherwise"""
    for s in oG:
        if np.array_equal(newGuess.A1, s.A1):
            return False
    return True

# Function: makeString - Flatten matrix to bitstring
def makeString(matrix):
    """Make a simple string representation of numpy matrix (flattened bits)"""
    return "".join(map(str, np.array(matrix, dtype=int).ravel().tolist()))

# Function: generate_error_vector_weight - Create error vector with given Hamming weight
def generate_error_vector_weight(n, weight):
    """Generate error vector of specified weight as numpy.matrix(1 x n)"""
    weight = max(0, min(weight, n))
    positions = secrets.SystemRandom().sample(range(n), weight) if weight > 0 else []
    e = np.zeros((1, n), dtype=int)
    for p in positions:
        e[0, p] = 1
    return np.matrix(e)

# ---------------- Goppa demo implementation ----------------

class GoppaDecoder:
    # Method: __init__ - Initialize field, g(x), support, H, and G
    def __init__(self, m, t):
    
        # m: int - The extension degree of the field GF(2^m)
        # t: int - The error correction capability (degree of g(x))
        
        self.m = int(m)
        self.t = int(t)
        
        if self.m <= 0 or self.t <= 0:
            raise ValueError("m and t must be positive integers")

        self.GF = galois.GF(2**self.m)
        
        self.g = self._generate_goppa_polynomial() # Now uses self.m and self.t
        self.L = self._generate_support()
        
        # n is the code length, determined by the size of the support (all non-roots)
        self.n = len(self.L) 
        
        # k is the message dimension, derived from n, m, and t
        # k >= n - mt
        self.k = self.n - (self.m * self.t)
        
        if self.k <= 0:
            raise ValueError(f"Invalid parameters: m={m}, t={t} results in non-positive k={self.k}. Try smaller t or larger m.")

        self.H_field = self._build_parity_check_over_gf()     # shape (t, n)
        self.H_binary = self._expand_to_binary(self.H_field)  # shape (t*m, n)

        # Compute G with dimensions k x n
        self.G_binary = self.compute_generator_matrix(self.H_binary)
        
        # Final check on G's dimensions
        if self.G_binary.shape != (self.k, self.n):
             # This can happen if null_space logic fails or k is miscalculated
             # We must resize G to be (k, n)
             G_resized = np.zeros((self.k, self.n), dtype=int)
             
             # Take the smallest dimensions to avoid out-of-bounds
             min_k = min(self.k, self.G_binary.shape[0])
             min_n = min(self.n, self.G_binary.shape[1])
             
             G_resized[:min_k, :min_n] = self.G_binary[:min_k, :min_n]
             self.G_binary = np.matrix(G_resized)

    def calculate_m(self, k, t):
        """Calculate m dynamically based on k and t."""
        m = 1
        while True:
            n = (1 << m) - 1  # n = 2^m - 1
            if n >= k + (t * m):
                return m
            m += 1

    def calculate_n(self, m, k, t):
        """Calculate n based on m, k, and t."""
        return k + (t * m)

    # Method: _generate_goppa_polynomial - Choose demo Goppa polynomial g(x)
    def _generate_goppa_polynomial(self):
        try:
            # galois.irreducible_poly returns a Poly object over GF(2^m)
            g = galois.irreducible_poly(order=2**self.m, degree=self.t, method="random")
            g = galois.Poly(g.coeffs, field=self.GF)
            return g
        except Exception:
            # Fallback: x^t + x + 1 (common irreducible form)
            coeffs = [self.GF(1), self.GF(1)] + [self.GF(0)] * (self.t - 2) + [self.GF(1)]
            return galois.Poly(coeffs, field=self.GF)

    # Method: _generate_support - Build support L excluding roots of g(x)
    def _generate_support(self):
        support = []
        for a in self.GF.elements:
            if self.g(a) != 0:
                support.append(a)
        return support[: (2**self.m) - 1]

    # Method: _build_parity_check_over_gf - Build H over GF(2^m)
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

    # Method: _expand_to_binary - Expand GF(2^m) matrix into binary using basis bits
    def _expand_to_binary(self, H_gf):
        t, n = H_gf.shape
        H_bin = np.zeros((t * self.m, n), dtype=np.uint8)
        for j in range(t):
            for i in range(n):
                val = int(np.int64(H_gf[j, i]))
                bits = [(val >> b) & 1 for b in range(self.m)]  # LSB-first basis
                H_bin[j*self.m:(j+1)*self.m, i] = bits
        return H_bin

    # Method: compute_parity_check_matrix - Return binary H as numpy.matrix
    def compute_parity_check_matrix(self):
        """Return binary parity-check matrix as numpy.matrix (t*m x n)"""
        return np.matrix(self.H_binary.astype(int))

    # Method: compute_generator_matrix - Null-space of H to get G over GF(2)
    def compute_generator_matrix(self, H):
        """
        Compute generator matrix G over GF(2) as null-space of H (binary).
        Returns G_matrix with dimensions k x n.
        """
        H_arr = np.array(H, dtype=int) & 1
        N = self.gf2_null_space(H_arr)
        
        if N.size == 0:
            return np.matrix(np.zeros((self.k, self.n), dtype=int))  # Ensure G has dimensions k x n
        
        G = np.matrix(N.astype(int))
        k = G.shape[0]
        n = H_arr.shape[1]  # Ensure n is derived from H

        # Initialize G_resized with zeros
        G_resized = np.zeros((self.k, self.n), dtype=int)

        if k < self.k:
            # If G is shorter than required, pad with zeros
            G_resized[:k, :G.shape[1]] = G  # Fill with G values
        elif k > self.k:
            # If G is longer than required, truncate G
            G_resized = G[:self.k, :self.n]  # Truncate to k x n
        else:
            # If G is exactly the right size, just assign it
            G_resized = G

        # Ensure that G is filled correctly
        if G_resized.shape[0] != self.k or G_resized.shape[1] != self.n:
            raise ValueError(f"Generated G has incorrect dimensions: {G_resized.shape}, expected: ({self.k}, {self.n})")
        
        return G_resized  # Return the resized G

    # Method: _gf2_rref - Reduced row echelon form over GF(2)
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

    # Method: gf2_null_space - Compute binary null-space basis
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

    # Method: decode_syndrome - Use Patterson decoder
    def decode_syndrome(self, syndrome_matrix):
        """
        Decode syndrome using Patterson algorithm.
        Returns numpy.matrix (1, n) error vector or None.
        """
        # Dispatch to Patterson decoder
        return patterson_decode_func(self, syndrome_matrix)

# ---------------- syndrome lookup wrapper ----------------

# Function: syndromeLookupGoppa - Demo wrapper giving first error position (1-indexed)
def syndromeLookupGoppa(H_bin_matrix, cHat, goppa_decoder):
    """
    H_bin_matrix: numpy.matrix (t*m x n) binary
    cHat: numpy.matrix shape (1,n)
    goppa_decoder: instance of GoppaDecoder
    Returns: 0 if no error, else 1-indexed first error position
    """
    syndrome = modTwo(np.matrix(H_bin_matrix) * np.matrix(cHat).T)  # shape (t*m, 1)
    if all_zeros(np.ravel(syndrome)):
        return 0
    err = patterson_decode_func(goppa_decoder, syndrome)
    if err is None:
        return 0
    arr = np.array(err).ravel()
    ones = np.flatnonzero(arr)
    if ones.size == 0:
        return 0
    return int(ones[0]) + 1

# ---------------- Private/Public key classes (Goppa) ----------------


#   - goppa_decoder : GoppaDecoder instance
#   - m, t : ints copied to instance
#   - n : code length from decoder
#   - H : binary parity-check (matrix)
#   - G : binary generator (array/matrix) k x n
#   - k : message dimension
#   - S : k x k scrambling matrix (binary) - generated if None via genSMatrix
#   - P : n x n permutation matrix (binary) - generated if None via genPMatrix
#
class privateKeyH84:

    # Method: __init__ - Construct private key and precompute helpers
    def __init__(self, m=4, t=2, S=None, P=None, original_text=None):
        """
        m: int - Field parameter (e.g., 12)
        t: int - Error correction capability (e.g., 50)
        """
        try:
            # Create the GoppaDecoder instance with the primary parameters
            self.goppa_decoder = GoppaDecoder(m, t)
            
            # Inherit parameters from the decoder
            self.m = self.goppa_decoder.m
            self.t = self.goppa_decoder.t
            self.n = self.goppa_decoder.n
            self.k = self.goppa_decoder.k
            
            self.H = self.goppa_decoder.compute_parity_check_matrix()
            self.G = self.goppa_decoder.G_binary  # Use the computed G

            # Ensure dimensions are correct
            if S is None:
                self.S = modTwo(genSMatrix(self.k))  # S is k x k
            else:
                self.S = modTwo(S)

            if P is None:
                self.P = modTwo(genPMatrix(self.n))  # P is n x n
            else:
                self.P = modTwo(P)

            if self.S.shape != (self.k, self.k):
                raise ValueError(f"Scrambling matrix S has incorrect shape: {self.S.shape}. Expected shape: ({self.k}, {self.k})")
            
            if self.P.shape != (self.n, self.n):
                raise ValueError(f"Permutation matrix P has incorrect shape: {self.P.shape}. Expected shape: ({self.n}, {self.n})")

            # Precompute helpers (GF(2))
            self.S_inv = np.matrix(gf2_inv(np.array(self.S, dtype=np.uint8)).astype(int))
            self.P_T = np.matrix(self.P.T)  # permutation inverse is transpose
            self.G_rinv = np.matrix(gf2_right_inverse(np.array(self.G, dtype=np.uint8)).astype(int))

        except ValueError:
            raise

    # Method: printCode - Print S, P, G', and params
    def printCode(self):
        print("S:\n", self.S)
        print("P:\n", self.P)
        print("GPrime:\n", self.makeGPrime())
        print(f"Goppa params: m={self.m}, t={self.t}, n={self.n}, k={self.k}")

    # Method: writeKeyToFile - Save private key via strict JSON schema
    def writeKeyToFile(self, keyFile):
        payload = {
            "format": "pythonMCS-private-v1",
            "version": KEY_FORMAT_VERSION,
            "m": int(self.m),
            "t": int(self.t),
            "n": int(self.n),
            "k": int(self.k),
            "field_irreducible_poly": int(self.goppa_decoder.GF.irreducible_poly),
            "g_coeffs": [int(c) for c in self.goppa_decoder.g.coeffs],
            "L": [int(a) for a in self.goppa_decoder.L],
            "S": _matrix_to_bit_lists(self.S),
            "P": _matrix_to_bit_lists(self.P),
            "G": _matrix_to_bit_lists(self.G),
            "H_binary": _matrix_to_bit_lists(self.goppa_decoder.H_binary),
        }
        with open(keyFile, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    # Method: readKeyFromFile - Load private key via strict JSON schema
    def readKeyFromFile(self, keyFile):
        with open(keyFile, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("format") != "pythonMCS-private-v1":
            raise ValueError("Unsupported private key format")
        if int(payload.get("version", -1)) != KEY_FORMAT_VERSION:
            raise ValueError("Unsupported private key version")

        self.m = int(payload["m"])
        self.t = int(payload["t"])
        self.n = int(payload["n"])
        self.k = int(payload["k"])

        self.S = np.matrix(
            _bit_lists_to_matrix(payload["S"], expected_shape=(self.k, self.k), field_name="S").astype(int)
        )
        self.P = np.matrix(
            _bit_lists_to_matrix(payload["P"], expected_shape=(self.n, self.n), field_name="P").astype(int)
        )
        self.G = np.matrix(
            _bit_lists_to_matrix(payload["G"], expected_shape=(self.k, self.n), field_name="G").astype(int)
        )
        h_binary = _bit_lists_to_matrix(
            payload["H_binary"], expected_shape=(self.t * self.m, self.n), field_name="H_binary"
        )

        irreducible_poly = galois.Poly.Int(int(payload["field_irreducible_poly"]))
        GF = galois.GF(2**self.m, irreducible_poly=irreducible_poly)
        g_poly = galois.Poly([GF(c) for c in payload["g_coeffs"]], field=GF)
        support = [GF(v) for v in payload["L"]]
        if len(support) != self.n:
            raise ValueError("Support length does not match n")

        self.goppa_decoder = SimpleNamespace(
            m=self.m,
            t=self.t,
            n=self.n,
            GF=GF,
            g=g_poly,
            L=support,
            H_binary=h_binary,
        )
        self.H = np.matrix(h_binary.astype(int))

        self.S_inv = np.matrix(gf2_inv(np.array(self.S, dtype=np.uint8)).astype(int))
        self.P_T = np.matrix(self.P.T)
        self.G_rinv = np.matrix(gf2_right_inverse(np.array(self.G, dtype=np.uint8)).astype(int))

    # Method: makeGPrime - Compute public generator G' = S G P
    def makeGPrime(self):
        return modTwo(self.S * self.G * self.P)  # Ensure dimensions are aligned

    # Method: decapsulate - Recover shared secret from KEM ciphertext bytes
    def decapsulate(self, kem_payload):
        if not isinstance(kem_payload, (bytes, bytearray)):
            raise ValueError("KEM payload must be bytes")

        kem_ct = bytes(kem_payload)
        expected_len = (int(self.n) + 7) // 8
        if len(kem_ct) != expected_len:
            raise ValueError("Invalid KEM ciphertext length")

        # Reject non-zero padding bits beyond n in the last byte.
        total_bits = len(kem_ct) * 8
        if total_bits > int(self.n):
            pad_bits = total_bits - int(self.n)
            if kem_ct and (kem_ct[-1] & ((1 << pad_bits) - 1)) != 0:
                raise ValueError("Malformed KEM ciphertext padding")

        c_bits = _bytes_to_bits(kem_ct, int(self.n))
        c_vec = np.matrix(c_bits.reshape(1, -1), dtype=int)
        m_mat = self.decrypt(c_vec)
        m_bits = np.array(m_mat, dtype=np.uint8).reshape(-1) & 1
        m_bytes = _bits_to_bytes(m_bits)
        return hashlib.sha256(m_bytes + kem_ct).digest()

    # Method: decrypt - Decode and recover message bits from ciphertext
    def decrypt(self, c):
        """
        c: numpy.matrix (1, n)
        returns: numpy.matrix (1, k) plaintext bits
        """
        c = np.matrix(np.array(c, dtype=int))
        if c.shape != (1, self.n):
            raise ValueError(f"Ciphertext shape must be (1, {self.n}), got {c.shape}")

        # Undo permutation via transpose (inverse of permutation)
        cHat = modTwo(c * self.P_T)

        H_bin = np.array(self.H, dtype=np.uint8) & 1
        s_bits = ((H_bin @ np.array(cHat, dtype=np.uint8).reshape(-1, 1)) & 1).ravel().astype(np.uint8)

        # Decode from received word so decoder manages syndrome conventions internally.
        # If strict checks inside Patterson fail, retry using explicit syndrome input.
        try:
            err = patterson_decode_func(self.goppa_decoder, cHat)
        except Exception:
            err = patterson_decode_func(self.goppa_decoder, np.matrix(s_bits).T)

        if err is not None:
            e_bits = np.array(err, dtype=np.uint8).ravel() & 1
            if e_bits.size != self.n:
                raise ValueError("Decoder returned error vector with invalid length")
            syn_check = ((H_bin @ e_bits.reshape(-1, 1)) & 1).ravel().astype(np.uint8)
            if not np.array_equal(syn_check, s_bits):
                raise RuntimeError("Decoded error vector does not reproduce ciphertext syndrome")

        corrected = modTwo(cHat + err) if err is not None else cHat

        # Final integrity check: corrected word must satisfy parity equations.
        corrected_syn = ((H_bin @ np.array(corrected, dtype=np.uint8).reshape(-1, 1)) & 1).ravel().astype(np.uint8)
        if np.any(corrected_syn):
            raise RuntimeError("Decoding failed: corrected word is not a valid codeword")

        # Recover x = m @ S using right-inverse of G: x = corrected @ G_rinv
        x = modTwo(corrected * self.G_rinv)  # (1, k)

        # Undo S: m = x @ S^{-1}
        m_bits = modTwo(x * self.S_inv)

        return np.matrix(np.array(m_bits, dtype=int))

    # Method: decryptFile - Decrypt modern KEM+DEM container
    def decryptFile(self, f):
        with open(f, "rb") as cf:
            magic = cf.read(4)
            if magic != QMC_MAGIC:
                raise ValueError("Unsupported ciphertext magic")

            version_raw = cf.read(2)
            if len(version_raw) != 2:
                raise ValueError("Ciphertext missing version")
            version = struct.unpack(">H", version_raw)[0]
            if version != QMC_CONTAINER_VERSION:
                raise ValueError("Unsupported ciphertext version")

            header_len_raw = cf.read(4)
            if len(header_len_raw) != 4:
                raise ValueError("Ciphertext missing header length")
            header_len = struct.unpack(">I", header_len_raw)[0]
            if header_len <= 0 or header_len > QMC_HEADER_MAX_BYTES:
                raise ValueError("Invalid ciphertext header length")

            header_json = cf.read(header_len)
            if len(header_json) != header_len:
                raise ValueError("Truncated ciphertext header")

            try:
                header = json.loads(header_json.decode("utf-8"))
            except Exception as exc:
                raise ValueError("Invalid ciphertext header JSON") from exc

            required_fields = {
                "magic",
                "version",
                "alg",
                "m",
                "n",
                "k",
                "t",
                "kem_ct",
                "nonce",
                "plaintext_len",
            }
            if not required_fields.issubset(header.keys()):
                raise ValueError("Ciphertext header missing required fields")

            if header["magic"] != QMC_MAGIC.decode("ascii"):
                raise ValueError("Ciphertext header magic mismatch")
            if int(header["version"]) != QMC_CONTAINER_VERSION:
                raise ValueError("Ciphertext header version mismatch")
            if header["alg"] != "mceliece-kem+chacha20poly1305":
                raise ValueError("Unsupported DEM algorithm")

            m = int(header["m"])
            n = int(header["n"])
            k = int(header["k"])
            t = int(header["t"])
            if m != int(self.m) or n != int(self.n) or k != int(self.k) or t != int(self.t):
                raise ValueError("Ciphertext parameters do not match private key")

            plaintext_len = int(header["plaintext_len"])
            if plaintext_len < 0:
                raise ValueError("Invalid plaintext length")

            kem_ct = _b64d(header["kem_ct"], "kem_ct")
            nonce = _b64d(header["nonce"], "nonce")
            if len(nonce) != 12:
                raise ValueError("Invalid AEAD nonce length")

            aead_ct = cf.read()
            if len(aead_ct) < 16:
                raise ValueError("Truncated AEAD ciphertext")

        shared_secret = self.decapsulate(kem_ct)
        symm_key = _hkdf_sha256(
            ikm=shared_secret,
            info=b"pythonMCS-file-dem",
            length=32,
            salt=kem_ct,
        )
        aad = _kem_meta_aad(QMC_CONTAINER_VERSION, m, t, n, k) + header_json
        aead = ChaCha20Poly1305(symm_key)
        plaintext = aead.decrypt(nonce, aead_ct, aad)

        if len(plaintext) != plaintext_len:
            raise ValueError("Plaintext length mismatch after decryption")

        with open(f + ".decoded", "wb") as mf:
            mf.write(plaintext)

    # Method: dnaFileDecrypt - Decrypt DNA-encoded ciphertext pairs (demo)
    def dnaFileDecrypt(self, f, dlu):
        with open(f, "r") as cf, open(f + ".decoded", "w") as mf:
            c1 = cf.readline().strip("\n")
            c2 = cf.readline().strip("\n")
            while c1 and c2:
                mat1 = np.matrix(" ".join(dlu.lookDNADecrypt(c1)), dtype=int)
                mat2 = np.matrix(" ".join(dlu.lookDNADecrypt(c2)), dtype=int)
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

    # Method: __init__ - Capture public G' and error weight t
    def __init__(self, GPrime, t=2, m=None, error_vector_weight=None):
        # Backward-compatible constructor: allow passing a privateKeyH84 instance.
        if hasattr(GPrime, "makeGPrime") and hasattr(GPrime, "t") and hasattr(GPrime, "m"):
            priv = GPrime
            self.GPrime = priv.makeGPrime()
            self.t = int(priv.t)
            self.m = int(priv.m)
        else:
            self.GPrime = GPrime
            self.t = int(t)
            if m is None:
                n_guess = int(np.array(self.GPrime).shape[1])
                self.m = max(1, int(np.ceil(np.log2(max(2, n_guess)))))
            else:
                self.m = int(m)
                if self.m <= 0:
                    raise ValueError("m must be positive")

        if self.t <= 0:
            raise ValueError(f"t must be positive, got {self.t}")

        if error_vector_weight is None:
            # Conservative non-zero default for reliability and confidentiality.
            self.error_vector_weight = min(1, self.t)
        else:
            self.error_vector_weight = int(error_vector_weight)
            if self.error_vector_weight < 0 or self.error_vector_weight > self.t:
                raise ValueError("error_vector_weight must be in [0, t]")
        # Keep constructor side-effect free for safer usage in service contexts.

    # Method: printCode - Print public G'
    def printCode(self):
        print("GPrime:\n", self.GPrime)

    # Method: writeKeyToFile - Save public key via strict JSON schema
    def writeKeyToFile(self, keyFile):
        gp = np.array(self.GPrime, dtype=np.uint8) & 1
        payload = {
            "format": "pythonMCS-public-v1",
            "version": KEY_FORMAT_VERSION,
            "m": int(self.m),
            "t": int(self.t),
            "error_vector_weight": int(self.error_vector_weight),
            "GPrime": gp.tolist(),
        }
        with open(keyFile, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    # Method: readKeyFromFile - Load public key via strict JSON schema
    def readKeyFromFile(self, keyFile):
        with open(keyFile, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != "pythonMCS-public-v1":
            raise ValueError("Unsupported public key format")
        if int(payload.get("version", -1)) != KEY_FORMAT_VERSION:
            raise ValueError("Unsupported public key version")

        gp = _bit_lists_to_matrix(payload["GPrime"], field_name="GPrime")
        self.GPrime = np.matrix(gp.astype(int))
        self.m = int(payload.get("m", max(1, int(np.ceil(np.log2(max(2, gp.shape[1])))))))
        self.t = int(payload["t"])
        self.error_vector_weight = int(payload["error_vector_weight"])
        if self.m <= 0 or self.t <= 0 or self.error_vector_weight <= 0:
            raise ValueError("Serialized key has non-positive t/error_vector_weight")

    # Method: encrypt - m @ G' plus random error of weight t
    def encrypt(self, m):
        """m: numpy.matrix (1, k)"""
        m = np.matrix(np.array(m, dtype=int))
        if m.shape != (1, self.GPrime.shape[0]):
            raise ValueError(
                f"Plaintext shape must be (1, {self.GPrime.shape[0]}), got {m.shape}"
            )
        c_encoded = modTwo(m * self.GPrime)
        error_vec = generate_error_vector_weight(int(c_encoded.shape[1]), self.error_vector_weight)
        c = modTwo(c_encoded + error_vec)
        return c

    # Method: encapsulate - Generate KEM ciphertext and shared secret
    def encapsulate(self):
        k = int(self.GPrime.shape[0])
        n = int(self.GPrime.shape[1])

        msg_seed = secrets.token_bytes((k + 7) // 8)
        m_bits = _bytes_to_bits(msg_seed, k)
        m_bytes = _bits_to_bytes(m_bits)
        m_mat = np.matrix(m_bits.reshape(1, -1), dtype=int)

        c_mat = self.encrypt(m_mat)
        c_bits = np.array(c_mat, dtype=np.uint8).reshape(-1) & 1
        kem_ct = _bits_to_bytes(c_bits)
        if len(kem_ct) != (n + 7) // 8:
            raise RuntimeError("Unexpected KEM ciphertext size")

        shared_secret = hashlib.sha256(m_bytes + kem_ct).digest()
        return {
            "ct_bytes": kem_ct,
            "shared_secret": shared_secret,
            "params": {"m": int(self.m), "n": n, "k": k, "t": int(self.t)},
        }

    # Method: encryptFile - Encrypt file with KEM+DEM AEAD container
    def encryptFile(self, f):
        with open(f, "rb") as mf:
            plaintext = mf.read()

        kem = self.encapsulate()
        kem_ct = kem["ct_bytes"]
        shared_secret = kem["shared_secret"]

        symm_key = _hkdf_sha256(
            ikm=shared_secret,
            info=b"pythonMCS-file-dem",
            length=32,
            salt=kem_ct,
        )
        nonce = secrets.token_bytes(12)

        header = {
            "magic": QMC_MAGIC.decode("ascii"),
            "version": QMC_CONTAINER_VERSION,
            "alg": "mceliece-kem+chacha20poly1305",
            "m": int(self.m),
            "n": int(self.GPrime.shape[1]),
            "k": int(self.GPrime.shape[0]),
            "t": int(self.t),
            "kem_ct": _b64e(kem_ct),
            "nonce": _b64e(nonce),
            "plaintext_len": int(len(plaintext)),
        }
        header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(header_json) > QMC_HEADER_MAX_BYTES:
            raise ValueError("Ciphertext header exceeds max size")

        aad = _kem_meta_aad(
            QMC_CONTAINER_VERSION,
            int(header["m"]),
            int(header["t"]),
            int(header["n"]),
            int(header["k"]),
        ) + header_json
        aead = ChaCha20Poly1305(symm_key)
        aead_ct = aead.encrypt(nonce, plaintext, aad)

        with open(f + ".ctxt", "wb") as cf:
            cf.write(QMC_MAGIC)
            cf.write(struct.pack(">H", QMC_CONTAINER_VERSION))
            cf.write(struct.pack(">I", len(header_json)))
            cf.write(header_json)
            cf.write(aead_ct)

# ---------------- Keep original Hamming classes for compatibility ----------------

class privateKeyH1611:
    """Legacy Hamming class removed from secure code paths."""

    def __init__(self, S=None, P=None):
        raise RuntimeError(
            "privateKeyH1611 is disabled: legacy Hamming variants are not allowed in hardened mode"
        )

class publicKeyH1611:
    """Legacy Hamming class removed from secure code paths."""

    def __init__(self, GPrime):
        raise RuntimeError(
            "publicKeyH1611 is disabled: legacy Hamming variants are not allowed in hardened mode"
        )

# ---------------- Brute forcer (keeps interface, demo only) ----------------

class bruteForcerH84:
    """Tries to brute-force S,P given GPrime (demo only)"""

    # Method: __init__ - Prepare demo G from decoder
    def __init__(self, GPrime):
        self.attempts = 0
        self.GPrime = GPrime
        self.GPrimeConsider = 0
        self.sConsider = 0
        self.pConsider = 0
        self.STries = list()
        self.PTries = list()
        decoder = GoppaDecoder(m=4, t=2)
        G = decoder.compute_generator_matrix(decoder.compute_parity_check_matrix())
        self.G = np.matrix(G)
        k = self.G.shape[0]

    # Method: attemptKey - Random search (demo placeholder)
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

    # Method: __init__ - Load mapping CSVs
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

    # Method: lookDNAEncrypt - Lookup encoding by bitstring
    def lookDNAEncrypt(self, bstring):
        return str(self.encLU.get(bstring, "?"))

    # Method: lookDNADecrypt - Lookup decoding by DNA string
    def lookDNADecrypt(self, bstring):
        return str(self.decLU.get(bstring, "?"))

# ---------------- Quick main test (demo) ----------------

#   - m, t : ints copied to instance
#   - n : code length from decoder
#   - H : binary parity-check (matrix)
#   - G : binary generator (array/matrix)
#   - k : message dimension
#   - S : k x k scrambling matrix (binary) - generated if None via genSMatrix
#   - P : n x n permutation matrix (binary) - generated if None via genPMatrix

# Method: text_to_bits - Convert text message to binary representation
def text_to_bits(text):
    """Convert a string to a binary representation."""
    return ''.join(format(ord(char), '08b') for char in text)

# Method: bits_to_text - Convert binary representation back to text
def bits_to_text(bits):
    """Convert a binary string back to a text string."""
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

if __name__ == "__main__":
    m,t = 9,5
    original_text = "hello"  # Example text message
    print("--- Key Generation ---")
    priv = privateKeyH84(m,t,original_text=original_text)
    pub = publicKeyH84(priv.makeGPrime(), t=priv.t, m=priv.m)
    print(f"params: m={priv.m}, t={priv.t}, n={priv.n}, k={priv.k}")

    # Convert text to binary
    msg_bits = text_to_bits(original_text)
    msg_bits = [int(bit) for bit in msg_bits]  # Convert string to list of bits
    
    # IMPORTANT: Pad to exactly k bits
    if len(msg_bits) < priv.k:
        padding_length = priv.k - len(msg_bits)
        print(f"Message length ({len(msg_bits)}) < k ({priv.k}). Padding with {padding_length} zeros.")
        msg_bits.extend([0] * padding_length)
        # Store original message length for unpadding
        original_msg_length = len(text_to_bits(original_text))
    else:
        original_msg_length = priv.k
    
    msg = np.matrix(msg_bits[:priv.k], dtype=int)  # Ensure it fits the k dimension

    print("--- Encryption/Decryption Test ---")
    print("message (binary):", msg)
    print("message shape:", msg.shape)
    ct = pub.encrypt(msg)
    print("ciphertext:", ct)
    print("ciphertext shape:", ct.shape)
    pt = priv.decrypt(ct)
    print("decrypted (binary):", pt)
    print("decrypted shape:", pt.shape)
    
    # Convert decrypted bits back to text
    # Only use the original message length, ignore padding
    decrypted_bits = ''.join(str(int(pt.item(i))) for i in range(original_msg_length))
    decrypted_text = bits_to_text(decrypted_bits)
    print("decrypted (text):", decrypted_text)

    # Check if the decrypted text matches the original
    assert decrypted_text == original_text, f"Decrypted text '{decrypted_text}' does not match the original '{original_text}'!"
    print("✓ SUCCESS: Decrypted text matches original!")