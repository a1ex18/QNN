"""Command-line utility to encrypt files with the hardened Goppa flow."""

import argparse

from goppa_COPY_mc_core import privateKeyH84, publicKeyH84

def print_usage():
	print ("\n\tUsage: python encrypt.py input_file [--m M] [--t T]\n")
	print ("\tSpits out three files:")
	print ("\t\tinput_file.priv - private key (JSON)")
	print ("\t\tinput_file.pub - public key (JSON)")
	print ("\t\tinput_file.ctxt - encrypted file\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("input_file", nargs="?", help="Input file path")
	parser.add_argument("--m", type=int, default=9, help="Field extension degree (default: 9)")
	parser.add_argument("--t", type=int, default=5, help="Error correction capability (default: 5)")
	args = parser.parse_args()

	if not args.input_file:
		print_usage()
		raise SystemExit(1)

	tPriv = privateKeyH84(m=args.m, t=args.t)
	tPub = publicKeyH84(tPriv.makeGPrime(), t=tPriv.t, m=tPriv.m)
	print("Encrypting...", tPub.encryptFile(args.input_file))
	tPriv.writeKeyToFile(str(args.input_file) + ".priv")
	tPub.writeKeyToFile(str(args.input_file) + ".pub")
	print("Done")


