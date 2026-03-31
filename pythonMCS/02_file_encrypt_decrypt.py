"""Hardened file-based encrypt/decrypt demo using encrypt.py/decrypt.py.

Flow:
1) Encrypt each dataset file using hardened `encrypt.py`.
2) Move generated `.ctxt`, `.priv`, and `.pub` artifacts into encoded mirror tree.
3) Decrypt each encoded `.ctxt` using `decrypt.py` and write outputs to decoded tree.
"""

import os
import shutil
import subprocess
import sys

from get_file_recursive import get_file, copy_structure


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCRYPT_SCRIPT = os.path.join(SCRIPT_DIR, "encrypt.py")
DECRYPT_SCRIPT = os.path.join(SCRIPT_DIR, "decrypt.py")

dataset_path = "/home/kali/Desktop/QNN/pythonMCS/dataset/COVID_19_Radiography_Dataset/"
encode_path = "/home/kali/Desktop/QNN/pythonMCS/dataset/encoded/"
decode_path = "/home/kali/Desktop/QNN/pythonMCS/dataset/decoded/"

os.makedirs(encode_path, exist_ok=True)
copy_structure(dataset_path, encode_path)

try:
    dataset_file_list = get_file(dataset_path)

    for d_file in dataset_file_list:
        print("d_file =", d_file)
        print("Encrypting via hardened encrypt.py")
        subprocess.run(
            [sys.executable, ENCRYPT_SCRIPT, d_file],
            check=True,
            cwd=SCRIPT_DIR,
        )

        rel_d_file_path = os.path.relpath(d_file, dataset_path)
        base_out = os.path.join(encode_path, rel_d_file_path)
        os.makedirs(os.path.dirname(base_out), exist_ok=True)

        # Keep key and ciphertext artifacts together in encoded tree.
        for suffix in (".ctxt", ".priv", ".pub"):
            src = d_file + suffix
            dst = base_out + suffix
            if os.path.exists(src):
                shutil.move(src, dst)

    print("<----------------------DECRYPTING---------------------->")

    encode_file_list = [f for f in get_file(encode_path) if f.endswith(".ctxt")]

    for e_file in encode_file_list:
        key_file = e_file[:-5] + ".priv"
        print("Decrypting via hardened decrypt.py")
        subprocess.run(
            [sys.executable, DECRYPT_SCRIPT, key_file, e_file],
            check=True,
            cwd=SCRIPT_DIR,
        )

        decode_file = e_file + ".decoded"
        decode_rel_path = os.path.relpath(e_file, encode_path)
        orig_rel_path = decode_rel_path.split(".ctxt", 1)[0]
        final_decode_file = os.path.join(decode_path, orig_rel_path)

        os.makedirs(os.path.dirname(final_decode_file), exist_ok=True)
        shutil.move(decode_file, final_decode_file)

except (FileNotFoundError, IOError, subprocess.CalledProcessError) as exc:
    print("Hardened pipeline failed:", exc)