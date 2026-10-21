#!/usr/bin/env python3
import sys
import tty
import termios
from http.server import HTTPServer, BaseHTTPRequestHandler

EXPECTED_IMAGE_BYTES = 64 * 64 * 3  # 12,288 bytes

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

def analyze_image_colors(image_bytes):
    """Parses 12,288 RGB bytes and prints color metrics."""
    if len(image_bytes) < EXPECTED_IMAGE_BYTES:
        print("  [WARN] Incomplete image buffer received.")
        return

    # De-interleave RGB channels using byte slicing
    reds = image_bytes[0::3]
    greens = image_bytes[1::3]
    blues = image_bytes[2::3]

    pixel_count = len(reds)  # 4096 pixels
    sum_r, sum_g, sum_b = sum(reds), sum(greens), sum(blues)
    avg_r, avg_g, avg_b = sum_r / pixel_count, sum_g / pixel_count, sum_b / pixel_count

    # Determine dominant color
    dominant = "Red" if avg_r >= avg_g and avg_r >= avg_b else ("Green" if avg_g >= avg_b else "Blue")

    print("┌──────────────────── Color Breakdown ────────────────────┐")
    print(f"│ Total Pixels : {pixel_count} (64x64)")
    print(f"│ Red Channel  : Avg {avg_r:5.1f} / 255  │ Total Sum: {sum_r:10,}")
    print(f"│ Green Channel: Avg {avg_g:5.1f} / 255  │ Total Sum: {sum_g:10,}")
    print(f"│ Blue Channel : Avg {avg_b:5.1f} / 255  │ Total Sum: {sum_b:10,}")
    print(f"│ Dominant     : {dominant}")
    print("└──────────────────────────────────────────────────────────┘")

class ArrowKeyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        raw_data = self.rfile.read(content_length)

        # Separate 12,288 RGB bytes from the trailing +1 extra byte
        image_bytes = raw_data[:EXPECTED_IMAGE_BYTES]
        extra_val = raw_data[EXPECTED_IMAGE_BYTES] if len(raw_data) > EXPECTED_IMAGE_BYTES else 0

        print(f"\n[!] POST Received: {len(raw_data)} bytes | Counter byte: {extra_val}")

        # Output color analysis
        analyze_image_colors(image_bytes)

        print("Press Arrow Key (↑ UP, ↓ DOWN, → RIGHT, ← LEFT): ", end="", flush=True)

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
        return  # Suppress default HTTP log prints

if __name__ == "__main__":
    server_address = ("0.0.0.0", 8000)
    httpd = HTTPServer(server_address, ArrowKeyHandler)
    print("=== ESP32 Controller & RGB Image Analyzer ===")
    print("Listening on port 8000. Waiting for ESP32...")
    httpd.serve_forever()
