"""
PC Server script to receive sensor data from Arduino (or simulated client),
run predictions, and send motor control signals back.
"""

import socket
import json
from pathlib import Path

HOST = "0.0.0.0"  # Listen on all local network interfaces
PORT = 5000       # Port to listen on
DEFAULT_MODEL = "bc_model.pt"
MODEL_DIR = Path("temp_models")


def get_model_path() -> Path:
    """Prompts the user in terminal to enter the model filename."""
    print("=" * 50)
    user_input = input(f"Enter model filename (default: {DEFAULT_MODEL}): ").strip()

    filename = user_input if user_input else DEFAULT_MODEL
    model_path = MODEL_DIR / filename if not Path(filename).is_absolute() and not Path(filename).exists() else Path(filename)

    if not model_path.exists():
        print(f"[WARNING] File '{model_path}' not found! Server will proceed, but make sure to train it.")
    else:
        print(f"[INFO] Using model file: {model_path}")

    print("=" * 50 + "\n")
    return model_path


def dummy_model_predict(distance: float, model_path: Path):
    """Placeholder prediction logic - replace with your actual PyTorch model inference."""
    if distance < 1.0:
        return {"steering_left": -1.0, "steering_right": 1.0}  # Hard turn
    else:
        return {"steering_left": 0.8, "steering_right": 0.8}    # Straight ahead


def start_server(model_path: Path):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow instant reuse of the port upon restart
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"[SERVER] PC Server listening on {HOST}:{PORT} (Active Model: {model_path.name})...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"\n[SERVER] Connection established from {client_address}")

        try:
            # Receive data (up to 1024 bytes)
            raw_data = client_socket.recv(1024).decode('utf-8')
            if not raw_data:
                continue

            # Parse incoming JSON payload
            sensor_data = json.loads(raw_data)
            print(f"[SERVER] Received sensor payload: {sensor_data}")

            # Extract sensor inputs
            dist = sensor_data.get("distance", 2.0)

            # Compute predictions using selected model path
            response_payload = dummy_model_predict(dist, model_path)
            print(f"[SERVER] Calculated response: {response_payload}")

            # Send response back to client
            client_socket.sendall(json.dumps(response_payload).encode('utf-8'))

        except Exception as e:
            print(f"[SERVER] Error handling client: {e}")
        finally:
            client_socket.close()


if __name__ == "__main__":
    # Prompt user for model name before starting server
    selected_model = get_model_path()
    start_server(selected_model)
