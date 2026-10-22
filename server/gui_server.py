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
        self.root.title("ESP32 UDP Streamer & Advanced Diagnostics Suite")
        self.root.geometry("1680x900")
        self.root.minsize(1400, 850)

        self.current_cmd = (0.0, 0.0)
        self.target_cmd = (0.0, 0.0)
        self.last_frame_time = time.perf_counter()
        self.incoming_queue = queue.Queue(maxsize=2)

        # Socket and address tracking
        self.udp_sock = None
        self.last_client_addr = None

        # Latency tracking state
        self.last_keypress_time = None
        self.last_active_time = time.time()
        self.pressed_keys = set()
        self.latency_timer_start = None

        # Diagnostic Counters
        self.diag_packets_received = 0
        self.diag_packets_dropped = 0
        self.diag_frames_assembled = 0
        self.diag_decode_errors = 0

        # Step-by-step telemetry metrics storage
        self.step_metrics = {
            "key_to_control_send": 0.0,
            "esp_capture": 0.0,
            "esp_udp_tx": 0.0,
            "video_transit": 0.0,
            "gui_render": 0.0,
            "total_latency": 0.0
        }

        # Moving Average state
        self.ema_alpha = 0.15
        self.fps_avg = None
        self.frame_time_avg = None

        self.frame_timestamps = deque()

        # Tkinter variables
        self.input_only_var = tk.BooleanVar(value=False)
        self.better_controls_var = tk.BooleanVar(value=True)
        self.flashlight_var = tk.BooleanVar(value=False)
        self.is_recording = False
        self.recording_buffer = []
        self.session_id = None
        self.temp_file_index = 0

        # Panel visibility state
        self.diag_panel_visible = True
        self.console_panel_visible = True

        os.makedirs("./temp_data", exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        # Count total historical frames in ./data/ at startup
        self.historical_dataset_frames = self._count_total_dataset_frames()

        self._configure_styles()
        self._build_ui()
        self._bind_keys()

        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_pynput_press,
            on_release=self._on_pynput_release
        )
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

        self.root.after(10, self._process_frames)
        self.root.after(20, self._update_motor_interpolation)
        self.root.after(5000, self._periodic_temp_save)
        self.root.after(100, self._update_inactivity_bar)
        self.root.after(1000, self._update_diagnostic_counters)

        self.udp_thread = threading.Thread(target=self._run_udp_server, daemon=True)
        self.udp_thread.start()

    def _count_total_dataset_frames(self):
        total = 0
        if os.path.exists("./data"):
            for f in os.listdir("./data"):
                if f.endswith(".pkl"):
                    path = os.path.join("./data", f)
                    try:
                        with open(path, "rb") as file:
                            data = pickle.load(file)
                            if isinstance(data, list):
                                total += len(data)
                    except Exception:
                        pass
        return total

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
        self.main_container = ttk.Frame(self.root, padding=16)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Canvas for Video Feed (640x640)
        self.canvas_container = ttk.LabelFrame(self.main_container, text=" Live Camera Feed (640x640) ", padding=8)
        self.canvas_container.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(self.canvas_container, width=640, height=640, bg="#111111", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        # 2. Telemetry & Controls Panel
        self.stats_frame = ttk.LabelFrame(self.main_container, text=" Telemetry & Controls ", padding=14)
        self.stats_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)

        self.lbl_fps = ttk.Label(self.stats_frame, text="FPS: 0 (Avg: 0)", font=("Helvetica", 11, "bold"), foreground="#27ae60")
        self.lbl_fps.pack(anchor="w", pady=4)

        self.lbl_latency = ttk.Label(self.stats_frame, text="Frame Time: -- ms (Avg: -- ms)", font=("Helvetica", 9))
        self.lbl_latency.pack(anchor="w", pady=4)

        self.lbl_timer_result = ttk.Label(
            self.stats_frame,
            text="Last Measured Delay: -- ms",
            font=("Helvetica", 9, "bold"),
            foreground="#d35400"
        )
        self.lbl_timer_result.pack(anchor="w", pady=4)

        self.lbl_dataset_frames = ttk.Label(
            self.stats_frame,
            text=f"Total Dataset Frames: {self.historical_dataset_frames}",
            font=("Helvetica", 9, "bold"),
            foreground="#2980b9"
        )
        self.lbl_dataset_frames.pack(anchor="w", pady=4)

        cmd_indicator_frame = ttk.Frame(self.stats_frame)
        cmd_indicator_frame.pack(anchor="w", pady=6)
        ttk.Label(cmd_indicator_frame, text="Active State: ", font=("Helvetica", 9)).pack(side=tk.LEFT)

        self.lbl_cmd_visual = ttk.Label(
            cmd_indicator_frame,
            text="STOP (0.00, 0.00)",
            font=("Helvetica", 10, "bold"),
            foreground="#7f8c8d"
        )
        self.lbl_cmd_visual.pack(side=tk.LEFT, padx=4)

        separator = ttk.Separator(self.stats_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=8)

        self.chk_better_controls = ttk.Checkbutton(
            self.stats_frame,
            text="Enable 'Better Controls' (Smooth & Mixed)",
            variable=self.better_controls_var
        )
        self.chk_better_controls.pack(anchor="w", pady=4)

        self.chk_flashlight = ttk.Checkbutton(
            self.stats_frame,
            text="ESP32-CAM Flashlight",
            variable=self.flashlight_var,
            command=self._on_flashlight_toggle
        )
        self.chk_flashlight.pack(anchor="w", pady=4)

        self.chk_input_mode = ttk.Checkbutton(
            self.stats_frame,
            text="Request frame on input only",
            variable=self.input_only_var,
            command=self._on_mode_toggle
        )
        self.chk_input_mode.pack(anchor="w", pady=4)

        self.chk_record_var = tk.BooleanVar(value=False)
        self.chk_record = ttk.Checkbutton(
            self.stats_frame,
            text="[ENTER] Toggle Recording",
            variable=self.chk_record_var,
            command=self.toggle_recording_from_ui
        )
        self.chk_record.pack(anchor="w", pady=4)

        btn_frame = ttk.Frame(self.stats_frame)
        btn_frame.pack(anchor="w", fill=tk.X, pady=(8, 4))

        self.btn_toggle_diag = ttk.Button(
            btn_frame,
            text="Hide Diagnostics",
            command=self._toggle_diagnostic_panel
        )
        self.btn_toggle_diag.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_toggle_console = ttk.Button(
            btn_frame,
            text="Hide Console",
            command=self._toggle_console_panel
        )
        self.btn_toggle_console.pack(side=tk.LEFT)

        inactivity_frame = ttk.Frame(self.stats_frame)
        inactivity_frame.pack(fill=tk.X, pady=(6, 4))

        ttk.Label(inactivity_frame, text="Inactivity Auto-Stop Timeout", font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.inactivity_bar = ttk.Progressbar(inactivity_frame, orient="horizontal", length=180, mode="determinate", maximum=100, style="Horizontal.TProgressbar")
        self.inactivity_bar.pack(fill=tk.X, pady=(3, 0))

        self.lbl_status = ttk.Label(self.stats_frame, text="Status: Listening (UDP)", font=("Helvetica", 9, "italic"), foreground="#e67e22")
        self.lbl_status.pack(anchor="w", pady=(10, 0))

        # 3. SECOND DIAGNOSTIC PANEL: Step-by-Step Latency & Packet Counters
        self.diag_frame = ttk.LabelFrame(self.main_container, text=" Step-by-Step Latency & Diagnostics ", padding=14)
        self.diag_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)

        ttk.Label(self.diag_frame, text="Pipeline Stage Timings", font=("Helvetica", 10, "bold"), foreground="#2980b9").pack(anchor="w", pady=(0, 6))

        self.lbl_step_1 = ttk.Label(self.diag_frame, text="1. Keypress -> Control Send: -- ms", font=("Consolas", 9))
        self.lbl_step_1.pack(anchor="w", pady=1)

        self.lbl_step_3 = ttk.Label(self.diag_frame, text="2. ESP32 Camera Capture & UART: -- ms", font=("Consolas", 9, "bold"), foreground="#8e44ad")
        self.lbl_step_3.pack(anchor="w", pady=1)

        self.lbl_step_4 = ttk.Label(self.diag_frame, text="3. ESP32 UDP Chunk Transmission: -- ms", font=("Consolas", 9, "bold"), foreground="#8e44ad")
        self.lbl_step_4.pack(anchor="w", pady=1)

        self.lbl_step_5 = ttk.Label(self.diag_frame, text="4. Video Network Transit & Reassemble: -- ms", font=("Consolas", 9))
        self.lbl_step_5.pack(anchor="w", pady=1)

        self.lbl_step_6 = ttk.Label(self.diag_frame, text="5. Python GUI Decode & Render: -- ms", font=("Consolas", 9))
        self.lbl_step_6.pack(anchor="w", pady=1)

        separator_diag = ttk.Separator(self.diag_frame, orient="horizontal")
        separator_diag.pack(fill=tk.X, pady=6)

        self.lbl_step_total = ttk.Label(self.diag_frame, text="TOTAL LATENCY: -- ms", font=("Helvetica", 10, "bold"), foreground="#c0392b")
        self.lbl_step_total.pack(anchor="w", pady=2)

        separator_diag2 = ttk.Separator(self.diag_frame, orient="horizontal")
        separator_diag2.pack(fill=tk.X, pady=6)

        ttk.Label(self.diag_frame, text="Live Packet Diagnostics", font=("Helvetica", 10, "bold"), foreground="#27ae60").pack(anchor="w", pady=(2, 4))
        self.lbl_diag_pkts = ttk.Label(self.diag_frame, text="Packets Rx: 0 | Dropped: 0", font=("Consolas", 8))
        self.lbl_diag_pkts.pack(anchor="w", pady=1)

        self.lbl_diag_frames = ttk.Label(self.diag_frame, text="Frames Assembled: 0 | Decode Err: 0", font=("Consolas", 8))
        self.lbl_diag_frames.pack(anchor="w", pady=1)

        self.lbl_diag_client = ttk.Label(self.diag_frame, text="Last Client Addr: None", font=("Consolas", 8), foreground="#7f8c8d")
        self.lbl_diag_client.pack(anchor="w", pady=(2, 0))

        # 4. CONSOLE PANEL (Right-hand side)
        self.console_frame = ttk.LabelFrame(self.main_container, text=" Diagnostic & Data Stream Console ", padding=10)
        self.console_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console = tk.Text(
            self.console_frame,
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

    def _update_layout(self):
        # Refresh packing layout seamlessly when toggling panels
        self.canvas_container.pack_forget()
        self.stats_frame.pack_forget()
        self.diag_frame.pack_forget()
        self.console_frame.pack_forget()

        self.canvas_container.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)
        self.stats_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)

        if self.diag_panel_visible:
            self.diag_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=False)

        if self.console_panel_visible:
            self.console_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _toggle_diagnostic_panel(self):
        if self.diag_panel_visible:
            self.btn_toggle_diag.config(text="Show Diagnostics")
            self.diag_panel_visible = False
        else:
            self.btn_toggle_diag.config(text="Hide Diagnostics")
            self.diag_panel_visible = True
        self._update_layout()

    def _toggle_console_panel(self):
        if self.console_panel_visible:
            self.btn_toggle_console.config(text="Show Console")
            self.console_panel_visible = False
        else:
            self.btn_toggle_console.config(text="Hide Console")
            self.console_panel_visible = True
        self._update_layout()

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {message}\n"
        print(full_msg, end="")

        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, full_msg)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def _update_diagnostic_counters(self):
        self.lbl_diag_pkts.config(text=f"Packets Rx: {self.diag_packets_received} | Dropped: {self.diag_packets_dropped}")
        self.lbl_diag_frames.config(text=f"Frames Assembled: {self.diag_frames_assembled} | Decode Err: {self.diag_decode_errors}")
        if self.last_client_addr:
            self.lbl_diag_client.config(text=f"Last Client Addr: {self.last_client_addr[0]}:{self.last_client_addr[1]}")

        total_current_frames = self.historical_dataset_frames + len(self.recording_buffer)
        self.lbl_dataset_frames.config(text=f"Total Dataset Frames: {total_current_frames}")

        self.root.after(1000, self._update_diagnostic_counters)

    def _bind_keys(self):
        self.root.bind("<Return>", self.toggle_recording_key)

    def _on_pynput_press(self, key):
        try:
            char = getattr(key, 'char', None)
            is_action_key = (
                key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right] or
                char in ['w', 'W', 's', 'S', 'a', 'A', 'd', 'D']
            )

            if is_action_key and self.latency_timer_start is None:
                self.latency_timer_start = time.perf_counter()
                self.step_metrics["key_to_control_send"] = 0.0
                self.log("[TIMER] Action key pressed -> Diagnostic Timer started.")

            if char == '-':
                if self.latency_timer_start is not None:
                    elapsed_ms = (time.perf_counter() - self.latency_timer_start) * 1000.0
                    self.log(f"[TIMER] '-' pressed! Measured total delay: {elapsed_ms:.1f} ms")
                    self.root.after(0, lambda: self.lbl_timer_result.config(text=f"Last Measured Delay: {elapsed_ms:.1f} ms"))
                    self.latency_timer_start = None
                else:
                    self.log("[TIMER] '-' pressed, but no action key timer was active.")

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
        up = "up" in self.pressed_keys
        down = "down" in self.pressed_keys
        left = "left" in self.pressed_keys
        right = "right" in self.pressed_keys

        if self.better_controls_var.get():
            throttle = 0.0
            steering = 0.0

            if up:
                throttle = 1.0
            elif down:
                throttle = -1.0

            if left:
                steering = -1.0
            elif right:
                steering = 1.0

            if throttle != 0.0:
                left_val = throttle + (steering * 0.65)
                right_val = throttle - (steering * 0.65)
            else:
                turn_speed = 0.5
                left_val = steering * turn_speed
                right_val = -steering * turn_speed

            target_left = max(min(left_val, 1.0), -1.0)
            target_right = max(min(right_val, 1.0), -1.0)
            self.target_cmd = (target_left, target_right)
        else:
            if up:
                cmd = (1.0, 1.0)
            elif down:
                cmd = (-1.0, -1.0)
            elif left:
                cmd = (-0.5, 0.5)
            elif right:
                cmd = (0.5, -0.5)
            else:
                cmd = (0.0, 0.0)
            self.target_cmd = cmd
            self.current_cmd = cmd
            self.set_command_visual(cmd)
            self._send_control_packet(cmd)

    def _update_motor_interpolation(self):
        if self.better_controls_var.get():
            lerp_rate = 0.25
            l_curr, r_curr = self.current_cmd
            l_target, r_target = self.target_cmd

            l_new = l_curr + (l_target - l_curr) * lerp_rate
            r_new = r_curr + (r_target - r_curr) * lerp_rate

            if abs(l_target - l_new) < 0.02:
                l_new = l_target
            if abs(r_target - r_new) < 0.02:
                r_new = r_target

            self.current_cmd = (l_new, r_new)
            self.set_command_visual(self.current_cmd)
            self._send_control_packet(self.current_cmd)

        self.root.after(25, self._update_motor_interpolation)

    def set_command_visual(self, cmd):
        l, r = cmd
        if abs(l) < 0.05 and abs(r) < 0.05:
            text, color = "STOP (0.00, 0.00)", "#7f8c8d"
        elif l > 0 and r > 0:
            if abs(l - r) < 0.1:
                text, color = f"FWD ({l:.2f}, {r:.2f})", "#27ae60"
            elif l > r:
                text, color = f"CURVE RIGHT ({l:.2f}, {r:.2f})", "#2980b9"
            else:
                text, color = f"CURVE LEFT ({l:.2f}, {r:.2f})", "#2980b9"
        elif l < 0 and r < 0:
            text, color = f"BACK ({l:.2f}, {r:.2f})", "#c0392b"
        else:
            text, color = f"PIVOT ({l:.2f}, {r:.2f})", "#e67e22"

        self.lbl_cmd_visual.config(text=text, foreground=color)
        if (abs(l) > 0.05 or abs(r) > 0.05):
            self.last_active_time = time.time()

    def _on_flashlight_toggle(self):
        self._send_control_packet(self.current_cmd)

    def _on_mode_toggle(self):
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
        if self.is_recording and abs(self.current_cmd[0]) < 0.05 and abs(self.current_cmd[1]) < 0.05:
            elapsed = time.time() - self.last_active_time
            progress = min(max((elapsed / 10.0) * 100.0, 0.0), 100.0)
            self.inactivity_bar['value'] = progress

            if elapsed > 10.0:
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
                self.historical_dataset_frames = self._count_total_dataset_frames()
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
            except Exception:
                pass

        final_filename = f"recording_{self.session_id}.pkl"
        final_path = os.path.join("./data", final_filename)

        try:
            with open(final_path, "wb") as f:
                pickle.dump(all_data, f)
            self.historical_dataset_frames = self._count_total_dataset_frames()
            self.log(f"[STORAGE] Dataset exported -> {final_path} ({len(all_data)} frames)")
        except Exception as e:
            self.log(f"[ERROR] Failed saving dataset: {e}")

        for tf in temp_files:
            try:
                os.remove(tf)
            except Exception:
                pass

    def _send_control_packet(self, cmd):
        if self.udp_sock and self.last_client_addr:
            try:
                t_send_start = time.perf_counter()
                mode_char = 'I' if self.input_only_var.get() else 'C'
                light_val = 1 if self.flashlight_var.get() else 0
                payload = f"{cmd[0]:.2f},{cmd[1]:.2f},{mode_char},{light_val}"
                self.udp_sock.sendto(payload.encode('utf-8'), self.last_client_addr)
                self.step_metrics["key_to_control_send"] = (time.perf_counter() - t_send_start) * 1000.0
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
            print(f"[PYTHON BIND ERROR] {e}")
            return

        pending_frames = {}

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                arrival_time = time.perf_counter()
                self.diag_packets_received += 1

                if len(data) < 12:
                    self.diag_packets_dropped += 1
                    continue

                frame_id = data[0]
                chunk_idx = data[1]
                total_chunks = data[2]
                counter = data[3]

                esp_capture_us = int.from_bytes(data[4:8], byteorder='little', signed=False)
                esp_tx_us = int.from_bytes(data[8:12], byteorder='little', signed=False)

                payload = data[12:]

                if frame_id not in pending_frames:
                    pending_frames[frame_id] = {
                        'chunks': {},
                        'total': total_chunks,
                        'counter': counter,
                        'esp_capture_us': esp_capture_us,
                        'esp_tx_us': esp_tx_us,
                        'first_chunk_arrival': arrival_time
                    }

                pending_frames[frame_id]['chunks'][chunk_idx] = payload

                if len(pending_frames[frame_id]['chunks']) == total_chunks:
                    full_frame = pending_frames.pop(frame_id)
                    jpeg_bytes = bytearray()
                    for idx in range(total_chunks):
                        if idx in full_frame['chunks']:
                            jpeg_bytes.extend(full_frame['chunks'][idx])
                        else:
                            self.log(f"[WARNING] Missing chunk {idx} for frame {frame_id}")

                    transit_time_ms = (arrival_time - full_frame['first_chunk_arrival']) * 1000.0
                    capture_ms = full_frame['esp_capture_us'] / 1000.0
                    tx_ms = full_frame['esp_tx_us'] / 1000.0

                    self.diag_frames_assembled += 1
                    now = time.perf_counter()
                    dt = now - self.last_frame_time
                    self.last_frame_time = now

                    if self.incoming_queue.full():
                        try:
                            self.incoming_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.incoming_queue.put((
                        bytes(jpeg_bytes),
                        full_frame['counter'],
                        dt,
                        capture_ms,
                        tx_ms,
                        transit_time_ms
                    ))
                    self.last_client_addr = (addr[0], CONTROL_PORT)

                    if not self.input_only_var.get():
                        self._send_control_packet(self.current_cmd)

                if len(pending_frames) > 15:
                    oldest_key = min(pending_frames.keys())
                    pending_frames.pop(oldest_key)
                    self.diag_packets_dropped += 1

            except Exception as e:
                print(f"[PYTHON UDP ERROR] {e}")

    def _process_frames(self):
        try:
            while not self.incoming_queue.empty():
                jpeg_bytes, counter, dt, capture_ms, tx_ms, transit_ms = self.incoming_queue.get_nowait()
                render_start = time.perf_counter()
                now = render_start

                # --- S & A DATA RECORDING (State & Action Only) ---
                if self.is_recording:
                    self.recording_buffer.append({
                        'image': jpeg_bytes,
                        'action': self.current_cmd
                    })
                # --------------------------------------------------

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

                avg_fps_str = f"{self.fps_avg:.0f}" if self.fps_avg is not None else "0"
                avg_frame_str = f"{self.frame_time_avg:.0f}" if self.frame_time_avg is not None else "--"

                self.lbl_fps.config(text=f"FPS: {round(fps)} (Avg: {avg_fps_str})")
                self.lbl_latency.config(text=f"Frame Time: {round(frame_ms)} ms (Avg: {avg_frame_str} ms)")

                try:
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    img.load()

                    img_view = img.resize((640, 640), Image.Resampling.BILINEAR)
                    self.tk_img = ImageTk.PhotoImage(img_view)
                    self.canvas.itemconfig(self.image_item, image=self.tk_img)

                    render_ms = (time.perf_counter() - render_start) * 1000.0

                    total_lat = (
                        self.step_metrics["key_to_control_send"] +
                        capture_ms +
                        tx_ms +
                        transit_ms +
                        render_ms
                    )

                    self.lbl_step_1.config(text=f"1. Keypress -> Control Send: {self.step_metrics['key_to_control_send']:.1f} ms")
                    self.lbl_step_3.config(text=f"2. ESP32 Camera Capture & UART: {capture_ms:.1f} ms")
                    self.lbl_step_4.config(text=f"3. ESP32 UDP Chunk Transmission: {tx_ms:.1f} ms")
                    self.lbl_step_5.config(text=f"4. Video Network Transit & Reassemble: {transit_ms:.1f} ms")
                    self.lbl_step_6.config(text=f"5. Python GUI Decode & Render: {render_ms:.1f} ms")
                    self.lbl_step_total.config(text=f"TOTAL LATENCY: {total_lat:.1f} ms")

                    self.lbl_status.config(text="Status: Streaming OK (UDP)", foreground="#27ae60")
                except Exception as e:
                    self.diag_decode_errors += 1
                    self.log(f"[PIL DECODE ERROR] {e}")
                    self.lbl_status.config(text="Status: Frame Decode Error", foreground="#e74c3c")
        finally:
            self.root.after(10, self._process_frames)

if __name__ == "__main__":
    root = tk.Tk()
    app = UDPStreamerApp(root)
    root.mainloop()
