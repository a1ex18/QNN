# Python McEliece Cryptosystem Experiments

Last updated: 2026-03-30

This folder contains both educational and research-oriented McEliece implementations.

## Hardened Path (Recommended)

Primary implementation: `goppa_COPY_mc_core.py`

CLI entry points:

```bash
python pythonMCS/encrypt.py <input_file> [--profile research-small|research-medium|research-large] [--m M --t T]
python pythonMCS/decrypt.py <private_key.json> <ciphertext.ctxt>
```

Features:

- Goppa-inspired private/public key flow (`G' = S G P`)
- JSON key serialization
- KEM+DEM container (`mceliece-kem+chacha20poly1305`)
- AEAD integrity protection via ChaCha20-Poly1305

## Legacy/Educational Path

Hamming-based scripts kept for demonstrations and attack timing experiments:

```bash
python pythonMCS/01_visual.py
python pythonMCS/02_file_encrypt_decrypt.py
python pythonMCS/03_private_key_crack.py
python pythonMCS/04_file_encrypt_crack_decrypt.py
python pythonMCS/05_H16_11_private_key_crack.py
```

## Dataset Workflow

Medical image assets used in experiments are under:

- `pythonMCS/dataset/COVID_19_Radiography_Dataset/`
- `pythonMCS/dataset/encoded/`
- `pythonMCS/dataset/decoded/`

## Dependencies

Install minimal dependencies for this component with:

```bash
pip install numpy galois cryptography
```

## Credits

- Original project reference: http://elibtronic.ca
- Author handle: http://twitter.com/elibtronic
