"""Command-line utility to decrypt files with the hardened Goppa flow."""

import argparse

from goppa_COPY_mc_core import privateKeyH84

def print_usage():
	print ("\n\tUsage: python decrypt.py private_key encrypted_file")
	print ("\tprivate_key: JSON file produced by encrypt.py")
	print ("\tencrypted_file: .ctxt file produced by encrypt.py")
	print ("\tOutputs encrypted_file.decoded\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("private_key", nargs="?", help="Private key JSON path")
	parser.add_argument("encrypted_file", nargs="?", help="Ciphertext file path")
	args = parser.parse_args()

	if not args.private_key or not args.encrypted_file:
		print_usage()
		raise SystemExit(1)

	tPriv = privateKeyH84()
	tPriv.readKeyFromFile(args.private_key)
	print("Decrypting...")
	tPriv.decryptFile(args.encrypted_file)
	print("Done")


