#!/usr/bin/env python3
import socket
import queue
import threading
import time
import io
import os
import random
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from collections import deque
from PIL import Image, ImageTk, ImageFile
from pynput import keyboard

ImageFile.LOAD_TRUNCATED_IMAGES = True

UDP_PORT = 8000
CONTROL_PORT = 8001

class UDPStreamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32 UDP Streamer & Data Recorder")
        self.root.geometry("760x820")
        self.root.minsize(700, 780)

        self.current_cmd = (0, 0)
        self.last_frame_time = time.perf_counter()
        self.incoming_queue = queue.Queue(maxsize=2)

        # Socket and address tracking for control responses
        self.udp_sock = None
        self.last_client_addr = None

        # Keypress timestamp tracker for KTF latency & inactivity timeout
        self.last_keypress_time = None
        self.last_active_time = time.time()
        self.pressed_keys = set()

        # Moving Average (Avg) state and factor alpha
        self.ema_alpha = 0.15
        self.fps_avg = None
        self.frame_time_avg = None
        self.key_latency_avg = None

        # Rolling window for smooth FPS based on actual frame count over time
        self.frame_timestamps = deque()

        # Tkinter variable for input-triggered frame request mode and recording
        self.input_only_var = tk.BooleanVar(value=False)
        self.is_recording = False
        self.recording_buffer = []
        self.session_id = None
        self.temp_file_index = 0

        # Ensure output directories exist
        os.makedirs("./temp_data", exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        self._configure_styles()
        self._build_ui()
        self._bind_keys()

        # Start pynput global/local keyboard listener for WASD + Arrow controls & Linux flickering fix
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_pynput_press,
            on_release=self._on_pynput_release
        )
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

        self.root.after(10, self._process_frames)
        self.root.after(5000, self._periodic_temp_save)
        self.root.after(100, self._update_inactivity_bar)

        self.udp_thread = threading.Thread(target=self._run_udp_server, daemon=True)
        self.udp_thread.start()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        bg_color = "#ffffff"
        text_color = "#111111"

        style.configure(".", background=bg_color, foreground=text_color, font=("Helvetica", 9), relief="flat")
        style.configure("TFrame", background=bg_color, relief="flat")
        style.configure("TLabelframe", background=bg_color, bordercolor="#e0e0e0", relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=bg_color, foreground="#555555", font=("Helvetica", 9, "bold"))
        style.configure("TButton", background="#f8f9fa", borderwidth=1, focusthickness=0, relief="flat")
        style.configure("TCheckbutton", background=bg_color, foreground=text_color)
        style.configure("Horizontal.TProgressbar", background="#27ae60", troughcolor="#f1f2f6", borderwidth=0)

    def _build_ui(self):
        main_container = ttk.Frame(self.root, padding=16)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top section: Canvas & Telemetry side-by-side
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        # Canvas for video feed
        canvas_container = ttk.LabelFrame(top_frame, text=" Live Camera Feed ", padding=8)
        canvas_container.pack(side=tk.LEFT, padx=(0, 12))

        self.canvas = tk.Canvas(canvas_container, width=384, height=384, bg="#111111", highlightthickness=0, bd=0)
        self.canvas.pack()
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        # Telemetry & Controls Panel
        stats_frame = ttk.LabelFrame(top_frame, text=" Telemetry & Controls ", padding=14)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_fps = ttk.Label(stats_frame, text="FPS: 0 (Avg: 0)", font=("Helvetica", 11, "bold"), foreground="#27ae60")
        self.lbl_fps.pack(anchor="w", pady=4)

        self.lbl_latency = ttk.Label(stats_frame, text="Frame Time: -- ms (Avg: -- ms)", font=("Helvetica", 9))
        self.lbl_latency.pack(anchor="w", pady=4)

        # Key-to-Frame label (conditionally packed when input-only mode is active)
        self.lbl_key_latency = ttk.Label(
            stats_frame,
            text="Key-to-Frame: -- ms (Avg: -- ms)",
            font=("Helvetica", 9, "bold"),
            foreground="#8e44ad"
        )

        # Visual indicator badge for active command tanh state
        cmd_indicator_frame = ttk.Frame(stats_frame)
        cmd_indicator_frame.pack(anchor="w", pady=6)
        ttk.Label(cmd_indicator_frame, text="Active State: ", font=("Helvetica", 9)).pack(side=tk.LEFT)

        self.lbl_cmd_visual = ttk.Label(
            cmd_indicator_frame,
            text="STOP (0, 0)",
            font=("Helvetica", 10, "bold"),
            foreground="#7f8c8d"
        )
        self.lbl_cmd_visual.pack(side=tk.LEFT, padx=4)

        separator = ttk.Separator(stats_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=8)

        # Interactive controls with keybind descriptor syntax
        self.chk_input_mode = ttk.Checkbutton(
            stats_frame,
            text="Request frame on input only",
            variable=self.input_only_var,
            command=self._on_mode_toggle
        )
        self.chk_input_mode.pack(anchor="w", pady=4)

        self.chk_record_var = tk.BooleanVar(value=False)
        self.chk_record = ttk.Checkbutton(
            stats_frame,
            text="[ENTER] Toggle Recording",
            variable=self.chk_record_var,
            command=self.toggle_recording_from_ui
        )
        self.chk_record.pack(anchor="w", pady=4)

        # Visual inactivity progress bar
        inactivity_frame = ttk.Frame(stats_frame)
        inactivity_frame.pack(fill=tk.X, pady=(10, 4))

        ttk.Label(inactivity_frame, text="Inactivity Auto-Stop Timeout", font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.inactivity_bar = ttk.Progressbar(inactivity_frame, orient="horizontal", length=180, mode="determinate", maximum=100, style="Horizontal.TProgressbar")
        self.inactivity_bar.pack(fill=tk.X, pady=(3, 0))

        self.lbl_status = ttk.Label(stats_frame, text="Status: Listening (UDP)", font=("Helvetica", 9, "italic"), foreground="#e67e22")
        self.lbl_status.pack(anchor="w", pady=(10, 0))

        # Bottom section: Dynamic Diagnostic Console (No scrollbar, expands to fill remaining space)
        console_frame = ttk.LabelFrame(main_container, text=" Diagnostic & Data Stream Console ", padding=10)
        console_frame.pack(fill=tk.BOTH, expand=True)

        self.console = tk.Text(
            console_frame,
            bg="#fafafa",
            fg="#222222",
            insertbackground="black",
            font=("Consolas", 9),
            highlightthickness=0,
            bd=0,
            wrap=tk.WORD
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {message}\n"
        print(full_msg, end="")

        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, full_msg)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def _bind_keys(self):
        self.root.bind("<Return>", self.toggle_recording_key)

    def _on_pynput_press(self, key):
        try:
            char = getattr(key, 'char', None)
            if key == keyboard.Key.up or char in ['w', 'W']:
                self.pressed_keys.add("up")
            elif key == keyboard.Key.down or char in ['s', 'S']:
                self.pressed_keys.add("down")
            elif key == keyboard.Key.left or char in ['a', 'A']:
                self.pressed_keys.add("left")
            elif key == keyboard.Key.right or char in ['d', 'D']:
                self.pressed_keys.add("right")

            self.root.after(0, self._evaluate_keys)
        except Exception:
            pass

    def _on_pynput_release(self, key):
        try:
            char = getattr(key, 'char', None)
            if key == keyboard.Key.up or char in ['w', 'W']:
                self.pressed_keys.discard("up")
            elif key == keyboard.Key.down or char in ['s', 'S']:
                self.pressed_keys.discard("down")
            elif key == keyboard.Key.left or char in ['a', 'A']:
                self.pressed_keys.discard("left")
            elif key == keyboard.Key.right or char in ['d', 'D']:
                self.pressed_keys.discard("right")

            self.root.after(0, self._evaluate_keys)
        except Exception:
            pass

    def _evaluate_keys(self):
        # Tanh activation mapping: fwd=(1,1), right=(1,-1), left=(-1,1), back=(-1,-1), idle=(0,0)
        if "up" in self.pressed_keys and "right" in self.pressed_keys:
            cmd = (1, 0) # or specific combination if desired, sticking to primary four directions:

        if "up" in self.pressed_keys:
            cmd = (1, 1)     # fwd
        elif "down" in self.pressed_keys:
            cmd = (-1, -1)   # back
        elif "left" in self.pressed_keys:
            cmd = (-1, 1)    # left
        elif "right" in self.pressed_keys:
            cmd = (1, -1)    # right
        else:
            cmd = (0, 0)     # idle

        self.set_command(cmd)

    def set_command(self, cmd):
        if self.current_cmd == cmd:
            return

        self.current_cmd = cmd

        cmd_map = {
            (0, 0): ("STOP (0, 0)", "#7f8c8d"),
            (1, 1): ("FWD (1, 1)", "#27ae60"),
            (-1, -1): ("BACK (-1, -1)", "#c0392b"),
            (-1, 1): ("LEFT (-1, 1)", "#2980b9"),
            (1, -1): ("RIGHT (1, -1)", "#2980b9")
        }
        text, color = cmd_map.get(cmd, ("STOP (0, 0)", "#7f8c8d"))
        self.lbl_cmd_visual.config(text=text, foreground=color)

        if cmd != (0, 0):
            self.last_active_time = time.time()

        if self.input_only_var.get():
            if self.last_keypress_time is None and cmd != (0, 0):
                self.last_keypress_time = time.perf_counter()
            self._send_control_packet(cmd)
        else:
            self._send_control_packet(cmd)

    def _on_mode_toggle(self):
        if not self.input_only_var.get():
            self.last_keypress_time = None
            self.lbl_key_latency.pack_forget()
        else:
            self.lbl_key_latency.pack(anchor="w", pady=4)
            avg_str = f"{self.key_latency_avg:.0f} ms" if self.key_latency_avg is not None else "-- ms"
            self.lbl_key_latency.config(text=f"Key-to-Frame: -- ms (Avg: {avg_str})")

        self._send_control_packet(self.current_cmd)

    def toggle_recording_key(self, event=None):
        self.is_recording = not self.is_recording
        self.chk_record_var.set(self.is_recording)
        self._handle_recording_state_change()

    def toggle_recording_from_ui(self):
        self.is_recording = self.chk_record_var.get()
        self._handle_recording_state_change()

    def _handle_recording_state_change(self):
        if self.is_recording:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_buffer = []
            self.temp_file_index = 0
            self.last_active_time = time.time()
            self.log(f"[ENTER] Recording session started. ID: {self.session_id}")
        else:
            self.log("[ENTER] Recording stopped by user. Exporting final dataset...")
            self._stop_and_save_recording()

    def _update_inactivity_bar(self):
        if self.is_recording and self.current_cmd == (0, 0):
            elapsed = time.time() - self.last_active_time
            progress = min(max((elapsed / 10.0) * 100.0, 0.0), 100.0)
            self.inactivity_bar['value'] = progress

            if elapsed > 10.0:
                self.log("[TIMEOUT] Inactivity limit reached (>10s). Auto-stopping recording.")
                self.is_recording = False
                self.chk_record_var.set(False)
                self._stop_and_save_recording()
                self.inactivity_bar['value'] = 0
        else:
            self.inactivity_bar['value'] = 0

        self.root.after(100, self._update_inactivity_bar)

    def _periodic_temp_save(self):
        if self.is_recording and self.recording_buffer:
            temp_filename = f"temp_{self.session_id}_{self.temp_file_index}.pkl"
            temp_path = os.path.join("./temp_data", temp_filename)
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(self.recording_buffer, f)
                self.log(f"[STORAGE] Flushed {len(self.recording_buffer)} frames to temp storage: {temp_path}")
                self.recording_buffer.clear()
                self.temp_file_index += 1
            except Exception as e:
                self.log(f"[ERROR] Failed writing temp file: {e}")

        self.root.after(5000, self._periodic_temp_save)

    def _stop_and_save_recording(self):
        if self.recording_buffer:
            self._periodic_temp_save()

        temp_files = [
            os.path.join("./temp_data", f)
            for f in os.listdir("./temp_data")
            if f.startswith(f"temp_{self.session_id}_")
        ]
        temp_files.sort()

        all_data = []
        for tf in temp_files:
            try:
                with open(tf, "rb") as f:
                    chunk = pickle.load(f)
                    all_data.extend(chunk)
            except Exception as e:
                self.log(f"[ERROR] Failed reading temp chunk {tf}: {e}")

        final_filename = f"recording_{self.session_id}.pkl"
        final_path = os.path.join("./data", final_filename)

        try:
            with open(final_path, "wb") as f:
                pickle.dump(all_data, f)
            self.log(f"[STORAGE] Final dataset exported successfully -> {final_path} ({len(all_data)} filtered frames)")
        except Exception as e:
            self.log(f"[ERROR] Failed saving final dataset: {e}")

        for tf in temp_files:
            try:
                os.remove(tf)
            except Exception:
                pass

    def _send_control_packet(self, cmd):
        if self.udp_sock and self.last_client_addr:
            try:
                mode_char = 'I' if self.input_only_var.get() else 'C'
                # cmd is a tuple like (1, 1), (-1, -1), etc.
                payload = f"{cmd[0]},{cmd[1]},{mode_char}"
                self.udp_sock.sendto(payload.encode('utf-8'), self.last_client_addr)
            except Exception as e:
                self.log(f"[CONTROL ERROR] {e}")

    def _calc_avg(self, current_avg, new_value):
        if current_avg is None:
            return new_value
        return (self.ema_alpha * new_value) + ((1.0 - self.ema_alpha) * current_avg)

    def _run_udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)

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
                    self.last_client_addr = (addr[0], CONTROL_PORT)

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

                if fps > 0:
                    self.fps_avg = self._calc_avg(self.fps_avg, fps)
                self.frame_time_avg = self._calc_avg(self.frame_time_avg, frame_ms)

                fps_avg_str = f"{self.fps_avg:.0f}" if self.fps_avg is not None else "0"
                frame_avg_str = f"{self.frame_time_avg:.0f}" if self.frame_time_avg is not None else "--"

                self.lbl_fps.config(text=f"FPS: {round(fps)} (Avg: {fps_avg_str})")
                self.lbl_latency.config(text=f"Frame Time: {round(frame_ms)} ms (Avg: {frame_avg_str} ms)")

                if key_latency_ms is not None and self.input_only_var.get():
                    self.key_latency_avg = self._calc_avg(self.key_latency_avg, key_latency_ms)
                    key_avg_str = f"{self.key_latency_avg:.0f}"
                    self.lbl_key_latency.config(text=f"Key-to-Frame: {round(key_latency_ms)} ms (Avg: {key_avg_str} ms)")

                if self.is_recording:
                    if self.current_cmd != (0, 0):
                        self.last_active_time = time.time()

                try:
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    img.load()

                    if self.is_recording:
                        should_record = True
                        if self.current_cmd == (0, 0):
                            if random.random() >= 0.3:
                                should_record = False

                        if should_record:
                            img_64 = img.resize((64, 64), Image.Resampling.BILINEAR).convert("RGB")
                            simulated_distance = random.randint(5, 300)

                            record_entry = {
                                "timestamp": time.time(),
                                "command": self.current_cmd,
                                "distance": simulated_distance,
                                "rgb_bytes": img_64.tobytes(),
                                "size": (64, 64)
                            }
                            self.recording_buffer.append(record_entry)
                            self.log(f"[RECORD] Cmd: {self.current_cmd} -> Dist: {simulated_distance}cm")

                    img_view = img.resize((384, 384), Image.Resampling.NEAREST)
                    self.tk_img = ImageTk.PhotoImage(img_view)
                    self.canvas.itemconfig(self.image_item, image=self.tk_img)
                    self.lbl_status.config(text="Status: Streaming OK (UDP)", foreground="#27ae60")
                except Exception as e:
                    self.log(f"[PIL DECODE ERROR] {e}")
                    self.lbl_status.config(text="Status: Frame Decode Error", foreground="#e74c3c")
        finally:
            self.root.after(10, self._process_frames)

if __name__ == "__main__":
    root = tk.Tk()
    app = UDPStreamerApp(root)
    root.mainloop()
