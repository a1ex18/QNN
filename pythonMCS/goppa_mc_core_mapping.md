# goppa_mc_core.py Function and Variable Mapping

Last updated: 2026-03-30

This document summarizes major functions, classes, and principal variables in `goppa_mc_core.py`.

## Top-Level Helper Functions

### genSMatrix(k)
- Purpose: generate a random invertible k×k binary matrix (numpy.matrix).
- Inputs: `k` (int)
- Returns: `sMaybe` (np.matrix k×k)
- Variables:
  - `k` — requested dimension.
  - `sMaybe` — candidate random 0/1 matrix; returned once invertible.

### genPMatrix(n, keep=False)
- Purpose: produce permutation matrix n×n.
- Inputs: `n` (int), `keep` (bool)
- Returns: permutation as `np.matrix`.
- Variables:
  - `n` — matrix dimension.
  - `keep` — if True return identity.
  - `p` — identity ndarray (base).
  - `perm` — permuted ndarray returned as matrix.

### modTwo(C)
- Purpose: return numeric contents of `C` modulo 2.
- Inputs: `C` (array-like)
- Returns: array-like reduced mod 2.
- Variables:
  - `C` — input container.
  - `D` — temporary copy (created but not required to compute return).

### bitFlip(C, n)
- Purpose: flip one bit in a 1×n vector.
- Inputs: `C` (np.matrix or np.array shape (1,n)), `n` (int; 0=no flip, -1=random, >0 -> 1-indexed pos)
- Returns: same type as `C`, mutated with one bit flipped.
- Variables:
  - `arr` — ndarray copy of `C` used for mutation.
  - `is_matrix` — True if input was `np.matrix` (to preserve return type).
  - `idx` — computed 0-based index to flip.

### all_zeros(d)
- Purpose: test whether all flattened elements are zero.
- Inputs: `d` (iterable / array-like)

### checkOldGuesses(oG, newGuess)
- Purpose: compare `newGuess.A1` against `.A1` of objects in `oG`.
- Inputs: `oG`, `newGuess`.

### makeString(matrix)
- Purpose: flatten `matrix` into a string of bits.
- Inputs: `matrix` (numpy array/matrix)

### generate_error_vector_weight(n, weight)
- Purpose: produce a 1×n error vector with exactly `weight` ones at random positions.
- Inputs: `n`, `weight`
- Variables:
  - `positions` — list of chosen indices.
  - `e` — returned error vector as `np.matrix(1, n)`.


## Class: GoppaDecoder

Purpose: demo Goppa code builder and brute-force syndrome decoder (small sizes only).

### __init__(self, m=4, t=2)
- Instance attributes:
  - `self.m` — GF exponent.
  - `self.t` — Goppa degree / error capability.
  - `self.GF` — galois.GF(2**m).
  - `self.g` — galois.Poly Goppa polynomial.
  - `self.L` — support list (non-roots).
  - `self.n` — code length (`len(self.L)`).
  - `self.H_field` — parity-check over GF (t × n).
  - `self.H_binary` — expanded binary parity-check (t*m × n).
  - `self.G_binary` — binary generator (null-space rows).
  - `self.k` — message dimension (# rows of generator).

### _generate_goppa_polynomial(self)
- Small deterministic polynomial selection for demo; `coeffs` list used to construct `galois.Poly`.

### _generate_support(self)
- Build `support` list iterating `self.GF.elements`, excluding roots of `g`.

### _build_parity_check_over_gf(self)
- Build parity-check H over GF with H[j,i] = (alpha^j) / g(alpha).
- Variables: `denom`, `inv`, `pow_alpha`, `j`, `i`, `alpha`.

### _expand_to_binary(self, H_gf)
- Expand GF elements into m-bit columns (LSB-first).
- Variables: `val`, `bits`, `H_bin`.

### compute_parity_check_matrix(self)
- Returns `self.H_binary` as `np.matrix` dtype int.

### compute_generator_matrix(self, H)
- Compute generator by null-space over GF(2).
- Variables: `H_arr`, `N`, `G`, `k`.

### _gf2_rref(self, A)
- GF(2) row-reduction producing pivot columns.
- Variables: `m, n, r, pivots, c, rr, pivot`.

### gf2_null_space(self, A)
- Compute null-space basis rows; uses `_gf2_rref` and back-substitution.
- Variables: `R, pivots, free_cols, basis, v, f`.

### decode_syndrome(self, syndrome_matrix)
- Brute-force search for error patterns up to `min(self.t, 4)`.
- Variables: `syndrome`, `H_bin`, `n`, `max_search_weight`, `w`, `pos`, `e`, `syn`.


## Top-Level Function: syndromeLookupGoppa(H_bin_matrix, cHat, goppa_decoder)
- Purpose: compute syndrome and use decoder to find error; return 0 if none or 1-indexed first error position.
- Variables: `syndrome`, `err`, `arr`, `ones`.


## Class: privateKeyH84

Purpose: Goppa private key wrapper.

### __init__(self, S=None, P=None, m=4, t=2)
- Attributes: `goppa_decoder`, `m`, `t`, `n`, `H`, `G`, `k`, `S`, `P`.

### makeGPrime(self)
- Returns `modTwo(self.S * self.G * self.P)`.

### decrypt(self, c)
- Steps:
  - `cHat` — undo permutation using numeric inverse trick.
  - `error_pos` — syndromeLookupGoppa(...)
  - `m_mat` — corrected codeword via `bitFlip`.
  - `m_bits` — first `k` bits after undoing `S`.

### decryptFile(self, f) and dnaFileDecrypt(self, f, dlu)
- File-processing helpers; variables: `cb1`, `cb2`, `b1`, `b2`, `bits1`, `bits2`, `c1_m`, `c2_m`, `pad`, `d1`, `d2`, `m1`, `m2`, `combined`.


## Class: publicKeyH84

Purpose: public encryption interface using `GPrime`.

### __init__(self, GPrime, t=2)
- Attributes: `GPrime`, `t`, `error_vector_weight`.

### encrypt(self, m)
- Compute `c_encoded = modTwo(m * GPrime)`, generate `error_vec`, and return `modTwo(c_encoded + error_vec)`.

### encryptFile(self, f)
- Read bytes, map to message blocks of size `k`, encrypt, and write ciphertext bits as bytes.


## Legacy Hamming Classes
- `privateKeyH1611` and `publicKeyH1611` kept for compatibility with original Hamming (16,11) code.


## Class: bruteForcerH84

Purpose: naive brute-force of `S` and `P` given `GPrime` for demo-sized codes.
- Variables: `attempts`, `GPrime`, `GPrimeConsider`, `sConsider`, `pConsider`, `STries`, `PTries`, `G`, `k`.


## Class: lookupDNA(encryptFile, decryptFile)
- Purpose: load CSV mapping tables into `encLU` and `decLU` dictionaries.
- Methods: `lookDNAEncrypt(bstring)` and `lookDNADecrypt(bstring)` — return mapped string or "?".


## Main Demo (`if __name__ == "__main__")
- Creates `priv`, `pub`, a small `msg` of length `priv.k`, encrypts and decrypts, prints results.


---

Notes:

- This is a maintained developer reference, not an autogenerated dump.
- Keep this file aligned with `pythonMCS/goppa_mc_core.py` after refactors.
