"""
Simulates an Arduino Wi-Fi client sending sensor readings to localhost.
"""

import socket
import json
import time
import random

SERVER_IP = "127.0.0.1"  # Localhost
SERVER_PORT = 5000


def send_sensor_data(distance_value: float):
    try:
        # Create a socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))

        # Package data as JSON
        payload = {"distance": distance_value}
        json_data = json.dumps(payload)

        # Send data
        print(f"[ARDUINO] Sending distance: {distance_value:.2f} m")
        client_socket.sendall(json_data.encode('utf-8'))

        # Receive response from PC
        response = client_socket.recv(1024).decode('utf-8')
        control_signals = json.loads(response)

        print(f"[ARDUINO] Received control signals -> Left: {control_signals['steering_left']}, Right: {control_signals['steering_right']}\n")

        client_socket.close()

    except ConnectionRefusedError:
        print("[ARDUINO] Could not connect to PC server. Make sure pc_server.py is running!")


if __name__ == "__main__":
    print("Starting Arduino Simulation (Press Ctrl+C to stop)...\n")
    try:
        while True:
            # Generate a random distance reading between 0.2 and 3.0 meters
            fake_distance = random.uniform(0.2, 3.0)
            send_sensor_data(fake_distance)
            time.sleep(2)  # Send data every 2 seconds
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
