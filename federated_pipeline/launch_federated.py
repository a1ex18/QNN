import argparse
import subprocess
import sys
import time


# Function: parse_args - Helper routine for parse args logic.
# Parameters: none.
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument('--num_clients', type=int, default=2, help='Number of federated clients to launch')
    parser.add_argument('--rounds', type=int, default=3, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Local epochs per client per round')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--server_port', type=int, default=8080, help='Port for Flower server')
    parser.add_argument('--headless', action='store_true', help='Do not open new terminals for clients (run in background)')
    return parser.parse_args()


# Function: main - Helper routine for main logic.
# Parameters: none.
def main():
    args = parse_args()
    server_cmd = [sys.executable, 'federated_pipeline/server.py']
    client_cmd = [sys.executable, 'federated_pipeline/client.py']

    # Set environment variables for config
    import os
    os.environ['NUM_CLIENTS'] = str(args.num_clients)
    os.environ['ROUNDS'] = str(args.rounds)
    os.environ['LOCAL_EPOCHS'] = str(args.local_epochs)
    os.environ['BATCH_SIZE'] = str(args.batch_size)
    os.environ['IMAGE_SIZE'] = str(args.image_size)
    os.environ['SERVER_PORT'] = str(args.server_port)

    # Start server
    print(f"Launching Flower server on port {args.server_port}...")
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(3)  # Give server time to start

    # Start clients
    client_procs = []
    for i in range(args.num_clients):
        env = os.environ.copy()
        env['CLIENT_ID'] = f'client_{i+1}'
        print(f"Launching client {i+1}...")
        if args.headless:
            proc = subprocess.Popen(client_cmd, env=env)
        else:
            # Try to open in new terminal window
            proc = subprocess.Popen(['x-terminal-emulator', '-e', sys.executable, 'federated_pipeline/client.py'], env=env)
        client_procs.append(proc)
        time.sleep(1)

    # Wait for server to finish
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        server_proc.terminate()
        for proc in client_procs:
            proc.terminate()

if __name__ == "__main__":
    main()
