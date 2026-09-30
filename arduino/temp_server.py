#!/usr/bin/env python3
import sys
import tty
import termios
from http.server import HTTPServer, BaseHTTPRequestHandler

def get_arrow_key():
    """Captures raw arrow keypresses without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'A': return 'UP'
                if ch3 == 'B': return 'DOWN'
                if ch3 == 'C': return 'RIGHT'
                if ch3 == 'D': return 'LEFT'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None

class ArrowKeyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("\n[!] ESP32 checking in. Press an Arrow Key (↑ UP, ↓ DOWN, → RIGHT, ← LEFT): ", end="", flush=True)

        key_command = None
        payload = "00"  # Default fallback: both off

        while key_command is None:
            key_command = get_arrow_key()

        if key_command == 'UP':
            payload = "11"  # Both ON
            print("UP ↑ (Both ON)")
        elif key_command == 'DOWN':
            payload = "00"  # Both OFF
            print("DOWN ↓ (Both OFF)")
        elif key_command == 'RIGHT':
            payload = "01"  # A0 OFF, A1 ON
            print("RIGHT → (A1 ON, A0 OFF)")
        elif key_command == 'LEFT':
            payload = "10"  # A0 ON, A1 OFF
            print("LEFT ← (A0 ON, A1 OFF)")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format, *args):
        return  # Suppress HTTP access logging for clean terminal view

if __name__ == "__main__":
    server_address = ("0.0.0.0", 8000)
    httpd = HTTPServer(server_address, ArrowKeyHandler)
    print("=== ESP32 Arrow Key Controller ===")
    print("Listening on port 8000. Waiting for ESP32...")
    httpd.serve_forever()
