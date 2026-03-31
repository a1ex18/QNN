import numpy as np
import galois
from functools import lru_cache

# -------------------- Low-level GF(2) helpers --------------------

def _gf2_gauss_solve(A, b):
    """
    Solve A x = b over GF(2) with Gaussian elimination.
    A: uint8 (d, d), b: uint8 (d,)
    Returns x as uint8 (d,)
    Raises ValueError if singular (shouldn't happen for square-free g).
    """
    A = (A.astype(np.uint8) & 1).copy()
    b = (b.astype(np.uint8) & 1).copy()
    d = A.shape[0]
    row = 0
    pivots = []

    for col in range(d):
        # Find pivot
        piv = None
        for r in range(row, d):
            if A[r, col]:
                piv = r
                break
        if piv is None:
            continue
        # Swap rows
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
            b[row], b[piv] = b[piv], b[row]
        # Eliminate other rows
        for r in range(d):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
        pivots.append((row, col))
        row += 1
        if row == d:
            break

    # Check consistency
    # No need to explicitly back-substitute because we reduced to RREF
    # Extract solution by reading columns of pivots
    x = np.zeros(d, dtype=np.uint8)
    for r, c in pivots:
        # row r has a single 1 at column c (RREF); others eliminated
        x[c] = b[r] & 1

    # Verify
    if ((A @ x) & 1).astype(np.uint8).sum() != b.sum():
        # Last check to catch singular/ambiguous cases
        if not np.array_equal((A @ x) & 1, b & 1):
            raise ValueError("GF(2) solve failed or singular system.")
    return x & 1

# -------------------- Field and basis helpers --------------------

def _elem_to_bits(a, m):
    """Convert GF(2^m) element to m-bit little-endian vector."""
    val = int(a)
    return np.array([(val >> i) & 1 for i in range(m)], dtype=np.uint8)

def _bits_to_elem(bits, GF):
    """Convert m-bit little-endian vector to GF(2^m) element."""
    val = 0
    for i, b in enumerate(bits.tolist()):
        if b & 1:
            val |= (1 << i)
    return GF(val)

def _poly_coeffs_vec(poly, t):
    """
    Return coefficients as list length t over GF(2^m) in ascending degree.
    Poly may have smaller degree; pad with field zeros.
    """
    # galois.Poly exposes coefficients in descending degree order.
    coeffs = list(poly.coeffs)[::-1]
    if len(coeffs) < t:
        coeffs += [poly.field(0)] * (t - len(coeffs))
    else:
        coeffs = coeffs[:t]
    return coeffs

def _vec_GF2m_to_bits(vec_GF2m, m):
    """
    Concatenate m-bit representations of GF(2^m) vector (ascending index).
    Returns length t*m bit vector (uint8).
    """
    parts = [_elem_to_bits(a, m) for a in vec_GF2m]
    return np.concatenate(parts).astype(np.uint8)

def _bits_to_vec_GF2m(bits, GF, t, m):
    """
    Inverse of _vec_GF2m_to_bits. bits length is t*m (uint8).
    Returns list of length t of GF elements.
    """
    vec = []
    for i in range(t):
        chunk = bits[i*m:(i+1)*m]
        vec.append(_bits_to_elem(chunk, GF))
    return vec

# -------------------- Linear operators over GF(2) --------------------

@lru_cache(maxsize=None)
def _mul_matrix_cached(a_int, char_poly_int):
    """
    Return m x m binary matrix M(a) over GF(2) representing multiplication by a in GF(2^m).
    Args:
      a_int: integer representation of GF element a
      char_poly_int: integer for the field's irreducible polynomial; used as cache key
    """
    # Reconstruct field from char_poly_int and degree m
    # We'll obtain m from char_poly_int bit length - 1
    m = (char_poly_int.bit_length() - 1)
    GF = galois.GF(2**m, irreducible_poly=galois.Poly.Int(char_poly_int))
    a = GF(a_int)

    basis = [GF(1 << i) for i in range(m)]
    cols = []
    for e in basis:
        prod = a * e
        cols.append(_elem_to_bits(prod, m))
    # Columns correspond to basis vectors
    M = np.column_stack(cols).astype(np.uint8)
    return M
 
@lru_cache(maxsize=None)
def _square_matrix_cached(char_poly_int):
    """
    Return m x m binary matrix F such that bits(x^2) = F * bits(x) over GF(2).
    """
    m = (char_poly_int.bit_length() - 1)
    GF = galois.GF(2**m, irreducible_poly=galois.Poly.Int(char_poly_int))
    basis = [GF(1 << i) for i in range(m)]
    cols = []
    for e in basis:
        sq = e * e
        cols.append(_elem_to_bits(sq, m))
    F = np.column_stack(cols).astype(np.uint8)
    return F

def _build_squaring_system(GF, g_poly, t, m):
    """
    Build the (t*m) x (t*m) binary matrix A for the linear system:
       vec_bits(R)  -> vec_bits(R^2 mod g)
    Specifically Y = A X where X are bits of coeffs u_j of R(z) and
    Y are bits of coeffs of (sum_j u_j^2 * (z^j)^2 mod g).
    A is block matrix with blocks A[i,j] = M(q_{j,i}) @ F, where:
      - q_j(z) = (z^j)^2 mod g(z)
      - q_{j,i} is coeff of z^i in q_j(z)
      - F is the m x m matrix of squaring in GF(2^m)
      - M(a) is the m x m matrix of multiply by a in GF(2^m)
    """
    # Obtain characteristic polynomial integer for cache key
    char_poly = GF.irreducible_poly
    char_poly_int = int(char_poly)
    F = _square_matrix_cached(char_poly_int)

    Poly = galois.Poly
    z = Poly([1, 0], field=GF)
    # Precompute q_j(z) for each j
    q_list = []
    for j in range(t):
        qj = ((z ** j) ** 2) % g_poly
        q_list.append(_poly_coeffs_vec(qj, t))  # list of t GF elements

    # Assemble A as blocks
    A = np.zeros((t*m, t*m), dtype=np.uint8)
    for j in range(t):
        # Each source coeff u_j contributes to all target i via u_j^2 times q_{j,i}
        # Block column for j is stacked blocks for i
        for i in range(t):
            a = q_list[j][i]
            M_a = _mul_matrix_cached(int(a), char_poly_int)  # m x m
            block = (M_a @ F) & 1
            A[i*m:(i+1)*m, j*m:(j+1)*m] = block
    return A  # (t*m, t*m)

def _sqrt_poly_mod_via_linear_system(T_poly, GF, g_poly, t, m):
    """
    Compute R(z) such that R^2 = T (mod g) by solving a linear system over GF(2).
    Returns a polynomial R in GF[z] with deg < t.
    """
    A = _build_squaring_system(GF, g_poly, t, m)  # (t*m, t*m)
    T_coeffs = _poly_coeffs_vec(T_poly, t)
    b = _vec_GF2m_to_bits(T_coeffs, m)  # RHS length t*m
    x = _gf2_gauss_solve(A, b)  # length t*m
    R_coeffs = _bits_to_vec_GF2m(x, GF, t, m)
    Poly = galois.Poly
    R = Poly(R_coeffs, field=GF)
    return R

# -------------------- Algebraic Patterson core --------------------

def _poly_ext_gcd(a, b):
    """
    Extended GCD over GF(2^m)[z]. Returns (g, s, t) such that g = s*a + t*b.
    Normalizes g to monic.
    """
    Poly = galois.Poly
    GF = a.field
    zero = Poly([0], field=GF)
    one = Poly([1], field=GF)
    r0, r1 = a, b
    s0, s1 = one, zero
    t0, t1 = zero, one
    while r1 != zero:
        q, r = divmod(r0, r1)
        r0, r1 = r1, r
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    # Normalize to monic gcd
    if len(r0.coeffs) == 0:
        raise ValueError("Extended GCD returned zero polynomial; cannot normalize")
    lc = r0.coeffs[0]
    inv = lc ** -1
    r0 = r0 * inv
    s0 = s0 * inv
    t0 = t0 * inv
    return r0, s0, t0

def _syndrome_to_poly(syndrome_bits, GF, m, t):
    """
    Map (t*m) binary syndrome to polynomial S(z) with deg < t over GF(2^m),
    assuming bit-chunks map to GF int_repr on the field's polynomial basis.
    """
    Poly = galois.Poly
    sb = syndrome_bits.astype(np.uint8) & 1
    if sb.size < t * m:
        sb = np.pad(sb, (0, t*m - sb.size), mode="constant")
    coeffs = []
    for i in range(t):
        chunk = sb[i*m:(i+1)*m]
        val = 0
        for j, b in enumerate(chunk.tolist()):
            if b & 1:
                val |= (1 << j)
        coeffs.append(GF(val))
    return Poly(coeffs[::-1], field=GF)

def _eea_asymmetric(g_poly, R_poly, t):
    """
    Extended Euclidean Algorithm with Patterson asymmetric stopping:
      find (a, b) with deg(a) <= floor(t/2), deg(b) <= floor((t-1)/2)
      such that a ≡ b * R (mod g).
    """
    Poly = galois.Poly
    GF = g_poly.field
    a_bound = t // 2
    b_bound = (t - 1) // 2

    r_prev, r_curr = g_poly, R_poly
    v_prev, v_curr = Poly([0], field=GF), Poly([1], field=GF)

    while True:
        if r_curr.degree <= a_bound:
            return r_curr, v_curr
        q, rem = divmod(r_prev, r_curr)
        r_prev, r_curr = r_curr, rem
        v_prev, v_curr = v_curr, v_prev - q * v_curr
        if v_curr.degree > b_bound:
            return r_prev, v_prev

def _chien_search(sigma_poly, support):
    """
    Evaluate sigma on support and return indices where sigma(alpha) == 0.
    """
    idx = []
    for i, alpha in enumerate(support):
        try:
            if sigma_poly(alpha) == 0:
                idx.append(i)
        except Exception:
            pass
    return idx


def _syndrome_matches(H_bin, e_vec, s_bits):
    """Return True iff H_bin @ e^T equals s_bits over GF(2)."""
    syn = (H_bin @ e_vec.reshape(-1, 1)) & 1
    return np.array_equal(syn.ravel().astype(np.uint8), s_bits.astype(np.uint8))


# -------------------- Fallback (small t guaranteed) --------------------

def _fallback_min_weight(H_bin, s_bits, t):
    """
    Brute-force incremental solver for H e^T = s (GF(2)), up to weight t<=3.
    Returns error vector (1, n) or None.
    """
    s = (s_bits.astype(np.uint8) & 1)
    n = H_bin.shape[1]
    cols = [H_bin[:, i] for i in range(n)]

    if np.all(s == 0):
        return None
    for i in range(n):
        if np.array_equal(cols[i], s):
            e = np.zeros((1, n), dtype=np.uint8)
            e[0, i] = 1
            return np.matrix(e)
    if t >= 2:
        for i in range(n):
            r1 = cols[i] ^ s
            for j in range(i + 1, n):
                if np.array_equal(cols[j], r1):
                    e = np.zeros((1, n), dtype=np.uint8)
                    e[0, i] = 1
                    e[0, j] = 1
                    return np.matrix(e)
    if t >= 3:
        for i in range(n):
            vi = cols[i]
            for j in range(i + 1, n):
                vij = vi ^ cols[j]
                r2 = vij ^ s
                for k in range(j + 1, n):
                    if np.array_equal(cols[k], r2):
                        e = np.zeros((1, n), dtype=np.uint8)
                        e[0, i] = e[0, j] = e[0, k] = 1
                        return np.matrix(e)
    return None

# -------------------- Public entrypoint --------------------

def patterson_decode(self, syndrome_matrix):
    """
    Patterson decoder to replace GoppaDecoder.decode_syndrome.
    self: GoppaDecoder instance with attributes:
          m (int), t (int), GF (galois.FieldClass), g (galois.Poly over GF),
          L (list of GF elements), n (int), H_binary (optional cached parity-check)
    syndrome_matrix: numpy.matrix column (t*m, 1) or flat bits
    Returns: numpy.matrix (1, n) error vector over GF(2), or None if syndrome=0.
    """
    # Normalize input: either packed syndrome (t*m) or received word (n).
    inp = np.array(syndrome_matrix).ravel().astype(np.uint8) & 1
    if inp.size == 0 or np.all(inp == 0):
        return None

    GF = self.GF
    g_poly = self.g
    m = int(self.m)
    t = int(self.t)
    n = int(self.n)

    try:
        H_bin = np.array(self.H_binary, dtype=np.uint8) & 1
    except Exception:
        H_bin = np.array(self.compute_parity_check_matrix(), dtype=np.uint8) & 1

    if inp.size == n:
        s_bits = ((H_bin @ inp.reshape(-1, 1)) & 1).ravel().astype(np.uint8)
    elif inp.size == (t * m):
        s_bits = inp
    else:
        raise ValueError(
            f"Invalid input length {inp.size}; expected n={n} or t*m={t*m}"
        )

    def _decode_with_sbits(s_try):
        # Step 1: S(z) from binary syndrome
        S = _syndrome_to_poly(s_try, GF, m, t)
        Poly = galois.Poly
        if S == Poly([0], field=GF):
            return None

        # Step 2: T(z) = S(z)^(-1) + z  (mod g), since char-2 so +/- same
        gcd, U, _V = _poly_ext_gcd(S, g_poly)
        if gcd.degree != 0:
            raise ValueError("S not invertible mod g")
        S_inv = U % g_poly
        z = Poly([1, 0], field=GF)
        T = (S_inv + z) % g_poly

        # Step 3: Solve for R: R^2 = T (mod g) via linear system over GF(2)
        R = _sqrt_poly_mod_via_linear_system(T, GF, g_poly, t, m)

        # Step 4: EEA with asymmetric stopping -> a(z), b(z)
        a, b = _eea_asymmetric(g_poly, R, t)

        # Step 5: sigma(z) = a(z)^2 + z * b(z)^2
        sigma = (a * a + z * (b * b)) % g_poly

        # Step 6: Chien search over support L
        errs = _chien_search(sigma, self.L)
        if not errs:
            raise ValueError("No roots of sigma on support")

        # Step 7: Build error vector
        e = np.zeros((1, int(self.n)), dtype=np.uint8)
        for i in errs:
            if 0 <= i < int(self.n):
                e[0, i] = 1

        # Reject invalid locator outputs before returning.
        if int(np.sum(e)) > t:
            raise ValueError(f"Decoded error weight {int(np.sum(e))} exceeds t={t}")
        if not _syndrome_matches(H_bin, e.ravel(), s_bits):
            raise ValueError("Decoded error vector does not reproduce syndrome")

        return np.matrix(e, dtype=int)

    errors = []
    for s_try in (s_bits, s_bits[::-1]):
        try:
            out = _decode_with_sbits(s_try)
            if out is not None:
                return out
        except Exception as ex:
            errors.append(str(ex))

    # Bounded brute-force fallback for very small t only.
    if t <= 3:
        fb = _fallback_min_weight(H_bin, s_bits, t)
        if fb is not None:
            return fb

    msg = "; ".join(errors[-2:]) if errors else "unknown"
    raise RuntimeError(f"Patterson decoding failed: {msg}")
