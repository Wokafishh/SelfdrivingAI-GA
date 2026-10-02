#!/usr/bin/env python3
import socket
import queue
import threading
import time
import io
import tkinter as tk
from tkinter import ttk
from collections import deque
from PIL import Image, ImageTk, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

UDP_PORT = 8000
CONTROL_PORT = 8001

class UDPStreamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra-Fast ESP32 UDP Streamer (Debug Mode)")
        self.root.geometry("780x680")

        self.current_cmd = "00"
        self.last_frame_time = time.perf_counter()
        self.incoming_queue = queue.Queue(maxsize=2)

        # Socket and address tracking for control responses
        self.udp_sock = None
        self.last_client_addr = None

        # Keypress timestamp tracker for KTF latency
        self.last_keypress_time = None

        # Exponential Moving Average (EMA) state and factor alpha
        self.ema_alpha = 0.15
        self.fps_ema = None
        self.frame_time_ema = None
        self.key_latency_ema = None

        # Rolling window for smooth FPS based on actual frame count over time
        self.frame_timestamps = deque()

        # Tkinter variable for input-triggered frame request mode
        self.input_only_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_keys()

        self.root.after(10, self._process_frames)

        self.udp_thread = threading.Thread(target=self._run_udp_server, daemon=True)
        self.udp_thread.start()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(top_frame, width=384, height=384, bg="#111")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        stats_frame = ttk.LabelFrame(top_frame, text=" Live Performance ", padding=10)
        stats_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.lbl_fps = ttk.Label(stats_frame, text="FPS: 0.0 (EMA: 0.0)", font=("Helvetica", 12, "bold"), foreground="#27ae60")
        self.lbl_fps.pack(anchor="w", pady=5)

        self.lbl_latency = ttk.Label(stats_frame, text="Frame Time: -- ms (EMA: -- ms)", font=("Helvetica", 10))
        self.lbl_latency.pack(anchor="w", pady=5)

        self.lbl_key_latency = ttk.Label(
            stats_frame,
            text="Key-to-Frame: N/A (Continuous)",
            font=("Helvetica", 10, "bold"),
            foreground="#8e44ad"
        )
        self.lbl_key_latency.pack(anchor="w", pady=5)

        self.lbl_counter = ttk.Label(stats_frame, text="Counter: --", font=("Monospace", 10))
        self.lbl_counter.pack(anchor="w", pady=5)

        self.lbl_cmd = ttk.Label(stats_frame, text="Active Cmd: 00", font=("Helvetica", 11, "bold"), foreground="#2980b9")
        self.lbl_cmd.pack(anchor="w", pady=10)

        # Toggle switch with state change callback
        self.chk_input_mode = ttk.Checkbutton(
            stats_frame,
            text="Request frame on input only",
            variable=self.input_only_var,
            command=self._on_mode_toggle
        )
        self.chk_input_mode.pack(anchor="w", pady=5)

        self.lbl_status = ttk.Label(stats_frame, text="Status: Listening (UDP)...", font=("Helvetica", 10, "italic"), foreground="#e67e22")
        self.lbl_status.pack(anchor="w", pady=10)

    def _bind_keys(self):
        self.root.bind("<KeyPress-Up>", lambda e: self.set_command("11"))
        self.root.bind("<KeyPress-Down>", lambda e: self.set_command("00"))
        self.root.bind("<KeyPress-Left>", lambda e: self.set_command("10"))
        self.root.bind("<KeyPress-Right>", lambda e: self.set_command("01"))

    def set_command(self, cmd):
        self.current_cmd = cmd
        self.lbl_cmd.config(text=f"Active Cmd: {cmd}")

        # In "input only" mode, keypress explicitly triggers the control packet to request the next frame
        if self.input_only_var.get():
            # Lock to initial keypress; ignore key repeat overrides while frame is in-flight
            if self.last_keypress_time is None:
                self.last_keypress_time = time.perf_counter()
            self._send_control_packet(cmd)

    def _on_mode_toggle(self):
        if not self.input_only_var.get():
            self.last_keypress_time = None
            self.lbl_key_latency.config(text="Key-to-Frame: N/A (Continuous)")
        else:
            ema_str = f"{self.key_latency_ema:.1f} ms" if self.key_latency_ema is not None else "-- ms"
            self.lbl_key_latency.config(text=f"Key-to-Frame: -- ms (EMA: {ema_str})")

        # Notify ESP32 immediately when toggled between Continuous ('C') and Input-Only ('I') modes
        self._send_control_packet(self.current_cmd)

    def _send_control_packet(self, cmd):
        if self.udp_sock and self.last_client_addr:
            try:
                # Append 'I' for input-only / lockstep mode, 'C' for continuous streaming mode
                mode_char = 'I' if self.input_only_var.get() else 'C'
                payload = f"{cmd}{mode_char}"
                self.udp_sock.sendto(payload.encode('utf-8'), self.last_client_addr)
            except Exception as e:
                print(f"[PYTHON CONTROL ERROR] {e}")

    def _calc_ema(self, current_ema, new_value):
        if current_ema is None:
            return new_value
        return (self.ema_alpha * new_value) + ((1.0 - self.ema_alpha) * current_ema)

    def _run_udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)  # 2MB RX buffer

        try:
            sock.bind(("0.0.0.0", UDP_PORT))
            self.udp_sock = sock
            print(f"[PYTHON] UDP Listener bound successfully to port {UDP_PORT}")
        except Exception as e:
            print(f"[PYTHON BIND ERROR] Could not bind to port {UDP_PORT}: {e}")
            return

        pending_frames = {}
        pkts_received = 0

        while True:
            try:
                data, addr = sock.recvfrom(2048)
                pkts_received += 1

                if pkts_received % 100 == 1:
                    print(f"[PYTHON RECV] Receiving active UDP packets from {addr}...")

                if len(data) < 4:
                    continue

                frame_id, chunk_idx, total_chunks, counter = data[0], data[1], data[2], data[3]
                payload = data[4:]

                if frame_id not in pending_frames:
                    pending_frames[frame_id] = {
                        'chunks': {},
                        'total': total_chunks,
                        'counter': counter
                    }

                pending_frames[frame_id]['chunks'][chunk_idx] = payload

                if len(pending_frames[frame_id]['chunks']) == total_chunks:
                    full_frame = pending_frames.pop(frame_id)
                    jpeg_bytes = bytearray()
                    for idx in range(total_chunks):
                        jpeg_bytes.extend(full_frame['chunks'][idx])

                    now = time.perf_counter()
                    dt = now - self.last_frame_time
                    self.last_frame_time = now

                    # Measure round-trip keypress-to-frame latency
                    key_latency_ms = None
                    if self.last_keypress_time is not None:
                        key_latency_ms = (now - self.last_keypress_time) * 1000.0
                        self.last_keypress_time = None

                    if self.incoming_queue.full():
                        try:
                            self.incoming_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.incoming_queue.put((bytes(jpeg_bytes), full_frame['counter'], dt, key_latency_ms))

                    # Store sender destination for control packets
                    self.last_client_addr = (addr[0], CONTROL_PORT)

                    # Continuous mode auto-triggers the next frame packet on frame receipt
                    if not self.input_only_var.get():
                        self._send_control_packet(self.current_cmd)

                if len(pending_frames) > 10:
                    oldest_key = min(pending_frames.keys())
                    pending_frames.pop(oldest_key)

            except Exception as e:
                print(f"[PYTHON UDP ERROR] {e}")

    def _process_frames(self):
        try:
            while not self.incoming_queue.empty():
                jpeg_bytes, counter, dt, key_latency_ms = self.incoming_queue.get_nowait()
                now = time.perf_counter()

                self.frame_timestamps.append(now)

                while self.frame_timestamps and self.frame_timestamps[0] < now - 1.0:
                    self.frame_timestamps.popleft()

                if len(self.frame_timestamps) > 1:
                    elapsed = now - self.frame_timestamps[0]
                    fps = (len(self.frame_timestamps) - 1) / elapsed if elapsed > 0 else 0.0
                else:
                    fps = 0.0

                frame_ms = dt * 1000.0

                # Calculate EMAs
                if fps > 0:
                    self.fps_ema = self._calc_ema(self.fps_ema, fps)
                self.frame_time_ema = self._calc_ema(self.frame_time_ema, frame_ms)

                fps_ema_str = f"{self.fps_ema:.1f}" if self.fps_ema is not None else "0.0"
                frame_ema_str = f"{self.frame_time_ema:.1f}" if self.frame_time_ema is not None else "--"

                self.lbl_fps.config(text=f"FPS: {fps:.1f} (EMA: {fps_ema_str})")
                self.lbl_latency.config(text=f"Frame Time: {frame_ms:.1f} ms (EMA: {frame_ema_str} ms)")
                self.lbl_counter.config(text=f"Counter: {counter}")

                # Update Key-to-Frame EMA & Label
                if key_latency_ms is not None:
                    self.key_latency_ema = self._calc_ema(self.key_latency_ema, key_latency_ms)
                    key_ema_str = f"{self.key_latency_ema:.1f}"
                    self.lbl_key_latency.config(text=f"Key-to-Frame: {key_latency_ms:.1f} ms (EMA: {key_ema_str} ms)")

                try:
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    img.load()
                    img = img.resize((384, 384), Image.Resampling.NEAREST)
                    self.tk_img = ImageTk.PhotoImage(img)
                    self.canvas.itemconfig(self.image_item, image=self.tk_img)
                    self.lbl_status.config(text="Status: Streaming OK (UDP)", foreground="#27ae60")
                except Exception as e:
                    print(f"[PIL DECODE ERROR] {e}")
                    self.lbl_status.config(text="Status: Frame Decode Error", foreground="#e74c3c")
        finally:
            self.root.after(10, self._process_frames)

if __name__ == "__main__":
    root = tk.Tk()
    app = UDPStreamerApp(root)
    root.mainloop()
