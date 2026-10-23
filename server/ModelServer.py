#!/usr/bin/env python3
import socket
import queue
import threading
import time
import io
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageFile
from pynput import keyboard

# Add parent directory to sys.path so we can import arkitektur from ../arkitektur.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Optional PyTorch & Architecture import
try:
    import torch
    from arkitektur import DrivingNetwork, preprocess_image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Minimum AI output presets: 0 = off, 0.4 = lowest value that gets the motors moving, 1 = always full speed
MIN_OUTPUT_PRESETS = (0.0, 0.4, 1.0)
DEFAULT_MIN_OUTPUT = 0.4

UDP_PORT = 8000
CONTROL_PORT = 8001

class ESP32StreamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32 Streamer & DrivingNetwork Model Server")
        self.root.geometry("1150x740")
        self.root.minsize(980, 700)

        self.current_cmd = (0.0, 0.0)
        self.target_cmd = (0.0, 0.0)
        self.incoming_queue = queue.Queue(maxsize=2)

        # Socket and client tracking
        self.udp_sock = None
        self.last_client_addr = None
        self.pressed_keys = set()

        # Tkinter control variables
        self.manual_control_var = tk.BooleanVar(value=True)
        self.better_controls_var = tk.BooleanVar(value=True)
        self.flashlight_var = tk.BooleanVar(value=False)
        self.input_only_var = tk.BooleanVar(value=False)
        self.min_output_var = tk.DoubleVar(value=DEFAULT_MIN_OUTPUT)

        # PyTorch Model state & directory setup (targeting ../temp_models)
        self.model = None
        self.model_path = None

        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp_models"))
        os.makedirs(self.models_dir, exist_ok=True)

        self._configure_styles()
        self._build_ui()

        # Populate model dropdown at startup
        self._populate_model_dropdown()

        # Start pynput keyboard listener for manual control
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_pynput_press,
            on_release=self._on_pynput_release
        )
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

        # GUI loops
        self.root.after(10, self._process_frames)
        self.root.after(20, self._update_motor_interpolation)

        # Start UDP server thread
        self.udp_thread = threading.Thread(target=self._run_udp_server, daemon=True)
        self.udp_thread.start()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        bg_color = "#ffffff"
        text_color = "#111111"

        style.configure(".", background=bg_color, foreground=text_color, font=("Helvetica", 10), relief="flat")
        style.configure("TFrame", background=bg_color, relief="flat")
        style.configure("TLabelframe", background=bg_color, bordercolor="#dcdde1", relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=bg_color, foreground="#2f3640", font=("Helvetica", 10, "bold"))
        style.configure("TButton", background="#f5f6fa", borderwidth=1, focusthickness=0, relief="flat")
        style.configure("TCheckbutton", background=bg_color, foreground=text_color)

    def _build_ui(self):
        self.main_container = ttk.Frame(self.root, padding=16)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Left: Live Camera Feed Canvas (640x640)
        self.canvas_container = ttk.LabelFrame(self.main_container, text=" Live Camera Feed (ESP32-CAM) ", padding=8)
        self.canvas_container.pack(side=tk.LEFT, padx=(0, 16), fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(self.canvas_container, width=640, height=640, bg="#111111", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        # 2. Right: Controls & PyTorch Model Panel
        self.control_panel = ttk.LabelFrame(self.main_container, text=" Control & DrivingNetwork Server ", padding=16)
        self.control_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mode Selection
        ttk.Label(self.control_panel, text="Operation Mode", font=("Helvetica", 10, "bold"), foreground="#2980b9").pack(anchor="w", pady=(0, 6))

        self.chk_manual = ttk.Checkbutton(
            self.control_panel,
            text="Enable Manual Control (Keyboard)",
            variable=self.manual_control_var,
            command=self._on_mode_change
        )
        self.chk_manual.pack(anchor="w", pady=4)

        self.lbl_mode_status = ttk.Label(self.control_panel, text="Current Mode: Manual (Keyboard)", font=("Helvetica", 9, "italic"), foreground="#27ae60")
        self.lbl_mode_status.pack(anchor="w", pady=(0, 10))

        ttk.Separator(self.control_panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # PyTorch Model Selection & Dropdown
        ttk.Label(self.control_panel, text="PyTorch Model Selection", font=("Helvetica", 10, "bold"), foreground="#8e44ad").pack(anchor="w", pady=(0, 6))

        dropdown_frame = ttk.Frame(self.control_panel)
        dropdown_frame.pack(anchor="w", fill=tk.X, pady=4)

        self.model_combobox = ttk.Combobox(dropdown_frame, state="readonly", width=30)
        self.model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_dropdown_model_selected)

        self.btn_refresh = ttk.Button(dropdown_frame, text="Refresh", width=8, command=self._populate_model_dropdown)
        self.btn_refresh.pack(side=tk.LEFT)

        # Alternative Browse Button
        self.btn_load_custom = ttk.Button(self.control_panel, text="Browse Custom Model File...", command=self._load_custom_pytorch_model)
        self.btn_load_custom.pack(anchor="w", pady=4, fill=tk.X)

        self.lbl_model_info = ttk.Label(self.control_panel, text="Model: Not Loaded", font=("Consolas", 9), foreground="#7f8c8d")
        self.lbl_model_info.pack(anchor="w", pady=(2, 10))

        ttk.Separator(self.control_panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # Stream & Peripheral Settings
        ttk.Label(self.control_panel, text="Settings", font=("Helvetica", 10, "bold"), foreground="#2f3640").pack(anchor="w", pady=(0, 6))

        self.chk_better_controls = ttk.Checkbutton(
            self.control_panel,
            text="Smooth Motor Interpolation (Lerp)",
            variable=self.better_controls_var
        )
        self.chk_better_controls.pack(anchor="w", pady=4)

        self.chk_flashlight = ttk.Checkbutton(
            self.control_panel,
            text="ESP32-CAM Flashlight",
            variable=self.flashlight_var,
            command=self._send_current_command
        )
        self.chk_flashlight.pack(anchor="w", pady=4)

        self.chk_input_mode = ttk.Checkbutton(
            self.control_panel,
            text="Request frame on input only",
            variable=self.input_only_var,
            command=self._send_current_command
        )
        self.chk_input_mode.pack(anchor="w", pady=4)

        # The recordings used smoothed controls, so models output small values
        # (e.g. 0.1, -0.03) that are too weak to get the motors moving
        self.lbl_min_output = ttk.Label(self.control_panel)
        self.lbl_min_output.pack(anchor="w", pady=(8, 0))
        self.scale_min_output = ttk.Scale(
            self.control_panel,
            from_=MIN_OUTPUT_PRESETS[0],
            to=MIN_OUTPUT_PRESETS[-1],
            orient="horizontal",
            variable=self.min_output_var,
            command=self._on_min_output_change
        )
        self.scale_min_output.pack(anchor="w", fill=tk.X, pady=(2, 4))
        self._on_min_output_change()

        ttk.Separator(self.control_panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # Active Command Display
        ttk.Label(self.control_panel, text="Active Command Output", font=("Helvetica", 10, "bold"), foreground="#2f3640").pack(anchor="w", pady=(0, 4))

        cmd_frame = ttk.Frame(self.control_panel)
        cmd_frame.pack(anchor="w", pady=4)
        ttk.Label(cmd_frame, text="Motors (Left, Right): ").pack(side=tk.LEFT)
        self.lbl_cmd_visual = ttk.Label(cmd_frame, text="STOP (0.00, 0.00)", font=("Helvetica", 10, "bold"), foreground="#7f8c8d")
        self.lbl_cmd_visual.pack(side=tk.LEFT, padx=4)

        self.lbl_connection_status = ttk.Label(self.control_panel, text="Status: Waiting for ESP32 Connection...", font=("Helvetica", 9, "italic"), foreground="#e67e22")
        self.lbl_connection_status.pack(anchor="w", pady=(15, 0))

    def _populate_model_dropdown(self):
        model_items = []
        folder_display_name = "temp_models"

        if os.path.exists(self.models_dir):
            for file_name in sorted(os.listdir(self.models_dir)):
                if file_name.endswith(('.pt', '.pth')):
                    model_items.append(f"../temp_models/{file_name}")

        if model_items:
            self.model_combobox['values'] = model_items
            self.model_combobox.set(model_items[0])
            self._load_model_from_path(os.path.join(self.models_dir, model_items[0].split('/')[-1]), model_items[0])
        else:
            self.model_combobox['values'] = ["No models found in ../temp_models/"]
            self.model_combobox.set("No models found in ../temp_models/")

    def _on_dropdown_model_selected(self, event=None):
        selected_text = self.model_combobox.get()
        if not selected_text or "No models found" in selected_text:
            return

        filename = os.path.basename(selected_text)
        full_path = os.path.join(self.models_dir, filename)
        self._load_model_from_path(full_path, selected_text)

    def _load_custom_pytorch_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Custom PyTorch Model",
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if file_path:
            display_name = f"Custom Folder/{os.path.basename(file_path)}"
            self._load_model_from_path(file_path, display_name)

    def _load_model_from_path(self, file_path, display_label):
            if not TORCH_AVAILABLE:
                self.lbl_model_info.config(text="Error: torch/torchvision not installed", foreground="#c0392b")
                return

            try:
                # FIX: Load using standard Python pickle since weights were saved via pickle.dump()
                import pickle
                with open(file_path, "rb") as f:
                    loaded_obj = pickle.load(f)

                # RL models also contain a critic; only the actor ("actor.*" weights) drives the car
                if isinstance(loaded_obj, dict) and any(k.startswith("actor.") for k in loaded_obj):
                    loaded_obj = {k[len("actor."):]: v for k, v in loaded_obj.items() if k.startswith("actor.")}

                # If it's a state_dict (OrderedDict), instantiate DrivingNetwork and load weights
                if isinstance(loaded_obj, dict) or type(loaded_obj).__name__ == 'OrderedDict':
                    print(f"[INFO] Detected state_dict. Instantiating DrivingNetwork architecture...")
                    self.model = DrivingNetwork(in_channels=3)
                    self.model.load_state_dict(loaded_obj)
                else:
                    self.model = loaded_obj

                if hasattr(self.model, 'eval'):
                    self.model.eval()

                self.model_path = file_path
                self.lbl_model_info.config(text=f"Loaded: {display_label}", foreground="#27ae60")
                self.manual_control_var.set(False)
                self._on_mode_change()
            except Exception as e:
                self.lbl_model_info.config(text=f"Load Failed: {str(e)[:30]}", foreground="#c0392b")
                print(f"[MODEL LOAD ERROR] {e}")

    def _on_mode_change(self):
        if self.manual_control_var.get():
            self.lbl_mode_status.config(text="Current Mode: Manual (Keyboard)", foreground="#27ae60")
        else:
            if self.model is None:
                self.lbl_mode_status.config(text="Warning: AI Mode active, but no model loaded!", foreground="#c0392b")
            else:
                self.lbl_mode_status.config(text="Current Mode: DrivingNetwork Inference", foreground="#8e44ad")

    def _on_min_output_change(self, _value=None):
        """Snaps the slider to the nearest preset and updates its label."""
        value = self.min_output_var.get()
        snapped = min(MIN_OUTPUT_PRESETS, key=lambda preset: abs(preset - value))
        if snapped != value:
            self.min_output_var.set(snapped)
        text = "off" if snapped == 0 else f"faster wheel at least {snapped:g}"
        self.lbl_min_output.config(text=f"Minimum AI output: {text}")

    def _apply_min_output(self, cmd):
        """Scales (left, right) up so the faster wheel is at least the slider's value, keeping the
        ratio (and so the turn direction). Outputs that are already strong enough are left as they are."""
        minimum = self.min_output_var.get()
        left, right = cmd
        strongest = max(abs(left), abs(right))
        if minimum <= 0 or strongest < 1e-6 or strongest >= minimum:
            return (left, right)
        scale = minimum / strongest
        return (left * scale, right * scale)

    def _load_model_inference(self, pil_img):
            """Passes the incoming PIL image through DrivingNetwork with robust diagnostics."""
            if self.model is None or not TORCH_AVAILABLE:
                return (0.0, 0.0)

            try:
                # 1. Transform PIL image to [3, 64, 64] tensor (same preprocessing as training)
                tensor_img = preprocess_image(pil_img)

                # 2. CRITICAL FIX: Add batch dimension -> [1, 3, 64, 64] to satisfy Conv2d requirements
                tensor_img = tensor_img.unsqueeze(0)

                with torch.no_grad():
                    # 3. Support both get_action() or direct forward() calls safely
                    if hasattr(self.model, 'get_action'):
                        res = self.model.get_action(tensor_img)
                    else:
                        res = self.model(tensor_img)

                # 4. Extract scalar values from output tensors safely
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    left_tensor, right_tensor = res[0], res[1]
                    left_val = left_tensor.item() if hasattr(left_tensor, 'item') else float(left_tensor)
                    right_val = right_tensor.item() if hasattr(right_tensor, 'item') else float(right_tensor)
                else:
                    left_val, right_val = float(res[0]), float(res[1])

                # Live Diagnostic Console Log
                print(f"[AI INFERENCE DIAGNOSTIC] Pred Left: {left_val:+.4f} | Pred Right: {right_val:+.4f}")
                return (float(left_val), float(right_val))

            except Exception as e:
                import traceback
                print(f"\n[MODEL INFERENCE ERROR EXCEPTION]")
                traceback.print_exc()
                print("-" * 50)

            return (0.0, 0.0)

    def _on_pynput_press(self, key):
        if not self.manual_control_var.get():
            return
        try:
            char = getattr(key, 'char', None)
            if key == keyboard.Key.up or char in ['w', 'W']: self.pressed_keys.add("up")
            elif key == keyboard.Key.down or char in ['s', 'S']: self.pressed_keys.add("down")
            elif key == keyboard.Key.left or char in ['a', 'A']: self.pressed_keys.add("left")
            elif key == keyboard.Key.right or char in ['d', 'D']: self.pressed_keys.add("right")
            self.root.after(0, self._evaluate_keys)
        except Exception:
            pass

    def _on_pynput_release(self, key):
        if not self.manual_control_var.get():
            return
        try:
            char = getattr(key, 'char', None)
            if key == keyboard.Key.up or char in ['w', 'W']: self.pressed_keys.discard("up")
            elif key == keyboard.Key.down or char in ['s', 'S']: self.pressed_keys.discard("down")
            elif key == keyboard.Key.left or char in ['a', 'A']: self.pressed_keys.discard("left")
            elif key == keyboard.Key.right or char in ['d', 'D']: self.pressed_keys.discard("right")
            self.root.after(0, self._evaluate_keys)
        except Exception:
            pass

    def _evaluate_keys(self):
        if not self.manual_control_var.get():
            return

        up = "up" in self.pressed_keys
        down = "down" in self.pressed_keys
        left = "left" in self.pressed_keys
        right = "right" in self.pressed_keys

        if self.better_controls_var.get():
            throttle, steering = 0.0, 0.0
            if up: throttle = 1.0
            elif down: throttle = -1.0
            if left: steering = -1.0
            elif right: steering = 1.0

            if throttle != 0.0:
                left_val = throttle + (steering * 0.65)
                right_val = throttle - (steering * 0.65)
            else:
                turn_speed = 0.5
                left_val = steering * turn_speed
                right_val = -steering * turn_speed

            self.target_cmd = (
                max(min(left_val, 1.0), -1.0),
                max(min(right_val, 1.0), -1.0)
            )
        else:
            if up: cmd = (1.0, 1.0)
            elif down: cmd = (-1.0, -1.0)
            elif left: cmd = (-0.5, 0.5)
            elif right: cmd = (0.5, -0.5)
            else: cmd = (0.0, 0.0)
            self.target_cmd = cmd
            self.current_cmd = cmd
            self._update_command_visual(cmd)
            self._send_control_packet(cmd)

    def _update_motor_interpolation(self):
        if self.manual_control_var.get() and self.better_controls_var.get():
            lerp_rate = 0.25
            l_curr, r_curr = self.current_cmd
            l_target, r_target = self.target_cmd

            l_new = l_curr + (l_target - l_curr) * lerp_rate
            r_new = r_curr + (r_target - r_curr) * lerp_rate

            if abs(l_target - l_new) < 0.02: l_new = l_target
            if abs(r_target - r_new) < 0.02: r_new = r_target

            self.current_cmd = (l_new, r_new)
            self._update_command_visual(self.current_cmd)
            self._send_control_packet(self.current_cmd)

        self.root.after(25, self._update_motor_interpolation)

    def _update_command_visual(self, cmd):
        l, r = cmd
        if abs(l) < 0.05 and abs(r) < 0.05:
            text, color = "STOP (0.00, 0.00)", "#7f8c8d"
        elif l > 0 and r > 0:
            if abs(l - r) < 0.1: text, color = f"FWD ({l:.2f}, {r:.2f})", "#27ae60"
            elif l > r: text, color = f"CURVE RIGHT ({l:.2f}, {r:.2f})", "#2980b9"
            else: text, color = f"CURVE LEFT ({l:.2f}, {r:.2f})", "#2980b9"
        elif l < 0 and r < 0:
            text, color = f"BACK ({l:.2f}, {r:.2f})", "#c0392b"
        else:
            text, color = f"PIVOT ({l:.2f}, {r:.2f})", "#e67e22"

        self.lbl_cmd_visual.config(text=text, foreground=color)

    def _send_current_command(self):
        self._send_control_packet(self.current_cmd)

    def _send_control_packet(self, cmd):
        if self.udp_sock and self.last_client_addr:
            try:
                mode_char = 'I' if self.input_only_var.get() else 'C'
                light_val = 1 if self.flashlight_var.get() else 0
                payload = f"{cmd[0]:.2f},{cmd[1]:.2f},{mode_char},{light_val}"
                self.udp_sock.sendto(payload.encode('utf-8'), self.last_client_addr)
            except Exception as e:
                print(f"[CONTROL ERROR] {e}")

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
                if len(data) < 12:
                    continue

                frame_id = data[0]
                chunk_idx = data[1]
                total_chunks = data[2]
                payload = data[12:]

                if frame_id not in pending_frames:
                    pending_frames[frame_id] = {
                        'chunks': {},
                        'total': total_chunks
                    }

                pending_frames[frame_id]['chunks'][chunk_idx] = payload

                if len(pending_frames[frame_id]['chunks']) == total_chunks:
                    full_frame = pending_frames.pop(frame_id)
                    jpeg_bytes = bytearray()
                    for idx in range(total_chunks):
                        if idx in full_frame['chunks']:
                            jpeg_bytes.extend(full_frame['chunks'][idx])

                    if self.incoming_queue.full():
                        try:
                            self.incoming_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.incoming_queue.put(bytes(jpeg_bytes))
                    self.last_client_addr = (addr[0], CONTROL_PORT)

                    if not self.input_only_var.get():
                        self._send_control_packet(self.current_cmd)

                if len(pending_frames) > 15:
                    oldest_key = min(pending_frames.keys())
                    pending_frames.pop(oldest_key)

            except Exception as e:
                print(f"[PYTHON UDP ERROR] {e}")

    def _process_frames(self):
        try:
            while not self.incoming_queue.empty():
                jpeg_bytes = self.incoming_queue.get_nowait()

                img = Image.open(io.BytesIO(jpeg_bytes))
                img.load()

                # If AI Model mode is active, run inference through DrivingNetwork
                if not self.manual_control_var.get():
                    ai_cmd = self._load_model_inference(img)
                    ai_cmd = self._apply_min_output(ai_cmd)
                    self.current_cmd = ai_cmd
                    self._update_command_visual(ai_cmd)
                    self._send_control_packet(ai_cmd)

                img_view = img.resize((640, 640), Image.Resampling.BILINEAR)
                self.tk_img = ImageTk.PhotoImage(img_view)
                self.canvas.itemconfig(self.image_item, image=self.tk_img)

                self.lbl_connection_status.config(text="Status: Streaming Active (UDP)", foreground="#27ae60")
        except Exception as e:
            print(f"[DECODE ERROR] {e}")
            self.lbl_connection_status.config(text="Status: Frame Decode Error", foreground="#e74c3c")
        finally:
            self.root.after(10, self._process_frames)

if __name__ == "__main__":
    root = tk.Tk()
    app = ESP32StreamerApp(root)
    root.mainloop()
