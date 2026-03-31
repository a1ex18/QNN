"""
MIMO channel and signal helpers matching MATLAB usage:
build_data_to_signal, spat_mod_wt, signals_wt, beaming_bias.
"""
import numpy as np

# Function: build_data_to_signal - Helper routine for build data to signal logic.
# Parameters: `arr` is arr input value.
def build_data_to_signal(arr):
    """Flatten weight/bias array to 1D signal (float32)."""
    return np.asarray(arr, dtype=np.float32).ravel()

# Function: _awgn_signal - Helper routine for  awgn signal logic.
# Parameters: `signal` is signal input value; `snr_db` is snr db input value.
def _awgn_signal(signal, snr_db):
    """Add AWGN to real-valued signal for given SNR (dB). Signal power assumed 1."""
    sig_power = np.mean(signal ** 2)
    if sig_power <= 0:
        sig_power = 1.0
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape).astype(np.float32)
    return signal + noise

# Function: _quantize_to_bits - Helper routine for  quantize to bits logic.
# Parameters: `x` is input value for computation; `n_bits` is n bits input value.
def _quantize_to_bits(x, n_bits=8):
    """Quantize float array to n_bits per value; return bits (0/1), x_min, scale."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_min, x_max = x.min(), x.max()
    if x_max <= x_min:
        scale = 1.0
        x_norm = np.zeros_like(x)
    else:
        scale = float(x_max - x_min)
        x_norm = (x - x_min) / scale
    levels = 2 ** n_bits
    x_int = np.clip((x_norm * (levels - 1)).round().astype(np.int64), 0, levels - 1)
    x_uint = x_int.astype(np.uint8)  # 8 bits per value
    bits = np.unpackbits(x_uint.reshape(-1, 1), axis=1).ravel()
    return bits, float(x_min), scale, n_bits

# Function: _dequantize_from_bits - Helper routine for  dequantize from bits logic.
# Parameters: `bits` is bits input value; `shape` is shape input value; `x_min` is x min input value; `scale` is scale input value; `n_bits` is n bits input value.
def _dequantize_from_bits(bits, shape, x_min, scale, n_bits=8):
    """Reconstruct float array from bits (8 bits per value)."""
    size = int(np.prod(shape))
    levels = 2 ** n_bits
    n_bits_total = size * n_bits
    bits = np.asarray(bits[:n_bits_total], dtype=np.uint8)
    if len(bits) < n_bits_total:
        bits = np.resize(bits, n_bits_total)
    bits = bits.reshape(size, n_bits)
    x_int = np.zeros(size, dtype=np.int64)
    for i in range(n_bits):
        x_int = (x_int << 1) | bits[:, i]
    x_int = np.clip(x_int, 0, levels - 1)
    x_norm = x_int.astype(np.float64) / (levels - 1)
    out = x_norm * scale + x_min
    return out.astype(np.float32).reshape(shape)

# Function: spat_mod_wt - Helper routine for spat mod wt logic.
# Parameters: `Nt` is Nt input value; `Nr` is Nr input value; `SNRdB` is SNRdB input value; `w_signal` is w signal input value; `dist` is dist input value; `rep` is rep input value.
def spat_mod_wt(Nt, Nr, SNRdB, w_signal, dist, rep):
    """
    Simulate MIMO channel: repeat signal, add AWGN at SNRdB, decode.
    Returns decoded 1D signal (float32) and BER (based on quantized bits).
    """
    w_signal = np.asarray(w_signal, dtype=np.float32).ravel()
    n_bits = 8
    bits, x_min, scale, _ = _quantize_to_bits(w_signal, n_bits=n_bits)
    # repetition: repeat each bit `rep` times
    bits_rep = np.repeat(bits, rep)
    # simple AWGN on bipolar encoding: 0 -> -1, 1 -> +1
    sym = 2 * bits_rep.astype(np.float32) - 1
    received = _awgn_signal(sym, SNRdB)
    # decode: average over repetitions, threshold
    received = received.reshape(-1, rep).mean(axis=1)
    decoded_bits = (received >= 0).astype(np.uint8)
    # BER
    min_len = min(len(bits), len(decoded_bits))
    ber = np.mean(bits[:min_len] != decoded_bits[:min_len]) if min_len else 0.0
    # reconstruct float signal from decoded bits (use original shape length)
    len_orig = len(w_signal)
    try:
        decoded_signal = _dequantize_from_bits(
            decoded_bits, (len_orig,), x_min, scale, n_bits
        )
    except Exception:
        decoded_signal = np.zeros(len_orig, dtype=np.float32)
        decoded_signal[: min(len_orig, len(decoded_bits) * 8 // n_bits)] = w_signal[: min(len_orig, len(decoded_bits) * 8 // n_bits)]
    return decoded_signal.ravel(), float(ber)

# Function: signals_wt - Helper routine for signals wt logic.
# Parameters: `decoded_signal` is decoded signal input value; `w1_1` is w1 1 input value; `w2_1` is w2 1 input value; `neurons` is neurons input value.
def signals_wt(decoded_signal, w1_1, w2_1, neurons):
    """Split decoded 1D signal into two weight matrices with shapes of w1_1 and w2_1."""
    s1, s2 = w1_1.shape, w2_1.shape
    n1, n2 = int(np.prod(s1)), int(np.prod(s2))
    decoded = np.asarray(decoded_signal, dtype=np.float32).ravel()
    if len(decoded) < n1 + n2:
        decoded = np.resize(decoded, n1 + n2)
    wBeam1 = decoded[:n1].reshape(s1).astype(np.float32)
    wBeam2 = decoded[n1 : n1 + n2].reshape(s2).astype(np.float32)
    return wBeam1, wBeam2

# Function: beaming_bias - Helper routine for beaming bias logic.
# Parameters: `decoded_signal` is decoded signal input value; `b1_1` is b1 1 input value; `b1_2` is b1 2 input value; `neurons` is neurons input value.
def beaming_bias(decoded_signal, b1_1, b1_2, neurons):
    """Split decoded 1D signal into two bias vectors with shapes of b1_1 and b1_2."""
    s1, s2 = b1_1.shape, b1_2.shape
    n1, n2 = int(np.prod(s1)), int(np.prod(s2))
    decoded = np.asarray(decoded_signal, dtype=np.float32).ravel()
    if len(decoded) < n1 + n2:
        decoded = np.resize(decoded, n1 + n2)
    bBeam1 = decoded[:n1].reshape(s1).astype(np.float32)
    bBeam2 = decoded[n1 : n1 + n2].reshape(s2).astype(np.float32)
    return bBeam1, bBeam2
