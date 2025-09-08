"""main.py
Clean, fixed version (monolithic) restoring core features without the previous
accidental nested duplicate code. Keeps detection loop, master loading,
object classification ROI capture, and chatbot role editing. This should
compile without syntax errors.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2, time, os, json, threading, numpy as np
import glob, shutil, queue, sys, io

from utils import (
    load_config, ensure_dirs, estimate_orientation, read_yolo_labels,
    save_config, encrypt_api_key, decrypt_api_key, get_gemini_api_key, set_gemini_api_key
)
from inference import ModelWrapper, match_detections_to_master
from detection_engine import DetectionEngine
from annotator import Annotator

cfg = load_config()
ensure_dirs(cfg.get("data_dir","data"), cfg.get("models_dir","models"), cfg.get("master_dir","data/master"))

# locale
def _load_locale():
    path = os.path.join(cfg.get("locales_dir","locales"), cfg.get("default_locale","en") + ".json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[locale] fallback empty ({e})")
        return {}

LOCALE = _load_locale()
def t(key, fallback):
    return LOCALE.get(key, fallback)

class App:
    def __init__(self, root):
        """Initialize main application state and build primary UI."""
        self.root = root
        self.root.title(t('app_title','Webcam Master Checker'))
        # --- state ---
        self.device_idx = cfg.get('last_webcam_index', 0)
        self._frame = None  # last processed frame
        self.model = ModelWrapper()
        self.master = None  # {'image': path, 'labels': [...]} YOLO normalized
        # single-instance window refs
        self._webcam_settings_window = None
        self._object_classification_window = None
        self._chatbot_settings_window = None
        self.object_entries = []  # collected ROIs
        # build UI
        self._build_main()

    # --------- Helpers ---------
    def _get_friendly_webcam_names(self):
        """Attempt to obtain a list of friendly webcam names (Windows only).
        Returns list[str]; order may not match OpenCV index order but usually close.
        """
        names = []
        if os.name != 'nt':
            return names
        import subprocess
        queries = [
            ["wmic","path","Win32_PnPEntity","where","Service='usbvideo'","get","Name"],
            ["wmic","path","Win32_PnPEntity","where","PNPClass='Image'","get","Name"],
        ]
        seen = set()
        for cmd in queries:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if r.returncode != 0:
                    continue
                lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                if lines and lines[0].lower().startswith('name'):
                    lines = lines[1:]
                for l in lines:
                    if l and l not in seen:
                        names.append(l)
                        seen.add(l)
            except Exception:
                continue
        return names

    # ---------------- UI build -----------------
    def _build_main(self):
            # Main container row for video
            row0 = ttk.Frame(self.root)
            row0.grid(row=0, column=0, sticky='nsew')
            # Fixed-size main detection preview (letterboxed within preview_max_width/height)
            self._main_prev_w = cfg.get('preview_max_width', 960)
            self._main_prev_h = cfg.get('preview_max_height', 720)
            blank_main = ImageTk.PhotoImage(Image.fromarray(np.zeros((self._main_prev_h, self._main_prev_w, 3), dtype=np.uint8)))
            self.video_panel = tk.Label(row0, image=blank_main, width=self._main_prev_w, height=self._main_prev_h, relief='sunken')
            self.video_panel.image = blank_main
            self.video_panel.pack(padx=4, pady=4)
            # Menubar with Settings
            menubar = tk.Menu(self.root)
            settings_menu = tk.Menu(menubar, tearoff=0)
            settings_menu.add_command(label='Webcam Settings', command=self.open_webcam_settings)
            settings_menu.add_command(label='Object Classification', command=self.open_object_classification)
            settings_menu.add_command(label='ChatBot Settings', command=self.open_chatbot_settings)
            menubar.add_cascade(label='Settings', menu=settings_menu)
            self.root.config(menu=menubar)

            # Control buttons row
            row1 = ttk.Frame(self.root)
            row1.grid(row=1, column=0, sticky='ew')
            ttk.Button(row1, text=t('btn_start','Start Stream'), command=self.start).pack(side='left', padx=2)
            ttk.Button(row1, text=t('btn_stop','Stop'), command=self.stop).pack(side='left', padx=2)
            ttk.Button(row1, text=t('btn_capture','Capture Image'), command=self.capture_image).pack(side='left', padx=2)
            ttk.Button(row1, text=t('btn_annotate','Annotate Image'), command=self.open_annotator).pack(side='left', padx=2)
            ttk.Button(row1, text=t('btn_train','Train Model'), command=self.train_model).pack(side='left', padx=2)
            ttk.Button(row1, text='Test Model', command=self.test_model).pack(side='left', padx=2)

            row2 = ttk.Frame(self.root)
            row2.grid(row=2, column=0, sticky='ew')
            ttk.Button(row2, text=t('btn_load_master','Load Master Image'), command=self.load_master).pack(side='left', padx=2)
            ttk.Button(row2, text=t('btn_export','Export Results'), command=self.export_results).pack(side='left', padx=2)

            # status + feedback
            self.status_var = tk.StringVar(value=t('status_ready','Ready'))
            ttk.Label(self.root, textvariable=self.status_var).grid(row=3, column=0, sticky='w', padx=4)
            self.feedback = tk.Text(self.root, width=80, height=8, state='disabled', wrap='word')
            self.feedback.grid(row=4, column=0, padx=4, pady=4, sticky='nsew')
            self.root.grid_rowconfigure(4, weight=1)
            self.root.grid_columnconfigure(0, weight=1)

    # -------------- Webcam Settings (simplified) --------------
    def open_webcam_settings(self):
        if self._webcam_settings_window and self._webcam_settings_window.winfo_exists():
            self._webcam_settings_window.lift(); return
        win = tk.Toplevel(self.root)
        win.title('Webcam Settings')
        self._webcam_settings_window = win
        # Layout: Preview -> Device List -> (Discover/Set Default) -> (Start/Stop) -> (Cancel/OK)
        # Button style (larger)
        style = ttk.Style(win)
        try:
            style.configure('Wide.TButton', padding=(10, 6), font=('Segoe UI', 10))
        except Exception:
            pass
        # Preview (fixed size placeholder initially blank)
        self._preview_w = cfg.get('preview_max_width', 480)
        self._preview_h = cfg.get('preview_max_height', 360)
        preview_frame = ttk.Frame(win)
        preview_frame.pack(fill='x', padx=6, pady=(6,4))
        blank = np.zeros((self._preview_h, self._preview_w, 3), dtype=np.uint8)
        blank_img = ImageTk.PhotoImage(Image.fromarray(blank))
        self.webcam_preview_panel = tk.Label(preview_frame, image=blank_img, relief='sunken', width=self._preview_w, height=self._preview_h)
        self.webcam_preview_panel.image = blank_img
        self.webcam_preview_panel.pack()

        # Device list
        list_frame = ttk.Frame(win)
        list_frame.pack(fill='x', padx=6, pady=4)
        self.webcam_listbox = tk.Listbox(list_frame, width=50, height=6)
        self.webcam_listbox.pack(fill='x')

        # Row: Discover / Set Default
        row_discover = ttk.Frame(win)
        row_discover.pack(pady=(4,2))
        # center align by internal padding and pack
        ttk.Button(row_discover, text='Discover', command=self._discover_cams, style='Wide.TButton').pack(side='left', padx=8, pady=2)
        ttk.Button(row_discover, text='Set Default', command=self._select_webcam, style='Wide.TButton').pack(side='left', padx=8, pady=2)

        # Row: Start / Stop Preview
        row_preview = ttk.Frame(win)
        row_preview.pack(pady=(2,2))
        ttk.Button(row_preview, text='Start Preview', command=self._start_webcam_preview, style='Wide.TButton').pack(side='left', padx=8, pady=2)
        ttk.Button(row_preview, text='Stop Preview', command=self._stop_webcam_preview, style='Wide.TButton').pack(side='left', padx=8, pady=2)

        # Row: Cancel / OK
        row_close = ttk.Frame(win)
        row_close.pack(pady=(6,8))
        def _final_close():
            # Stop preview and also main capture to release camera fully
            try:
                self.stop()
            except Exception:
                pass
            self._stop_webcam_preview()
            if win.winfo_exists():
                win.destroy()
            self._webcam_settings_window = None
        def _cancel():
            _final_close()
        def _ok():
            sel = self.webcam_listbox.curselection()
            if sel:
                text = self.webcam_listbox.get(sel[0])
                try:
                    idx = int(text.split(':',1)[0])
                    self.device_idx = idx
                except Exception:
                    pass
            _final_close()
        ttk.Button(row_close, text='Cancel', command=_cancel, style='Wide.TButton').pack(side='left', padx=10, pady=4)
        ttk.Button(row_close, text='OK', command=_ok, style='Wide.TButton').pack(side='left', padx=10, pady=4)
        win.protocol('WM_DELETE_WINDOW', _cancel)
        self._discover_cams()

    def _discover_cams(self, max_test=5):
        if not hasattr(self, 'webcam_listbox'): return
        self.webcam_listbox.delete(0, tk.END)
        selected_line = None
        friendly = self._get_friendly_webcam_names()
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                name = friendly[i] if i < len(friendly) else f"Device {i}"
                # Truncate overly long names for UI clarity
                if len(name) > 48:
                    name = name[:45] + '...'
                line = f"{i}: {name}"
                self.webcam_listbox.insert(tk.END, line)
                if i == self.device_idx:
                    selected_line = self.webcam_listbox.size() - 1
                cap.release()
        if self.webcam_listbox.size() == 0:
            self.webcam_listbox.insert(tk.END, 'No devices found')
        else:
            # auto-select previously saved default if present
            try:
                if selected_line is not None:
                    self.webcam_listbox.select_set(selected_line)
                    self.webcam_listbox.see(selected_line)
            except Exception:
                pass

    def _select_webcam(self):
        if not hasattr(self, 'webcam_listbox'): return
        sel = self.webcam_listbox.curselection()
        if not sel:
            messagebox.showinfo('Select','Choose a webcam first')
            return
        text = self.webcam_listbox.get(sel[0])
        idx = text.split(':')[0].strip()
        try:
            idx = int(idx)
        except ValueError:
            messagebox.showerror('Error','Bad device entry')
            return
        self.device_idx = idx
        cfg['last_webcam_index'] = idx
        save_config(cfg)
        messagebox.showinfo('Saved', f'Default webcam set to {idx}')

    def _start_webcam_preview(self):
        # Determine desired device: selected listbox entry if available else saved default
        desired_idx = self.device_idx
        if hasattr(self, 'webcam_listbox'):
            sel = self.webcam_listbox.curselection()
            if sel:
                text = self.webcam_listbox.get(sel[0])
                try:
                    desired_idx = int(text.split(':',1)[0])
                except Exception:
                    pass
        # If already running on this device, do nothing
        if getattr(self, 'webcam_preview_running', False) and getattr(self, '_preview_device_idx', None) == desired_idx:
            return
        # Restart if running on different device
        if getattr(self, 'webcam_preview_running', False):
            self._stop_webcam_preview()
        cap = cv2.VideoCapture(desired_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror('Preview', f'Cannot open webcam {desired_idx}')
            return
        # Auto-select maximum resolution (heuristic) unless user already configured
        try:
            if not (cfg.get('camera_width') and cfg.get('camera_height')):
                for w,h in [
                    (3840,2160),(2560,1440),(2560,1080),(2560,960),(1920,1200),(1920,1080),
                    (1600,1200),(1600,900),(1536,864),(1440,900),(1366,768),(1280,1024),
                    (1280,960),(1280,800),(1280,720),(1024,768),(800,600),(800,480),
                    (720,480),(640,480),(640,360),(424,240),(320,240)
                ]:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    for _ in range(2):
                        cap.read()
                    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if abs(aw-w)<=32 and abs(ah-h)<=32:
                        cfg['camera_width'], cfg['camera_height'] = aw, ah
                        break
        except Exception:
            pass
        self.webcam_preview_cap = cap
        self._preview_device_idx = desired_idx
        self.webcam_preview_running = True
        self._webcam_preview_tick()

    def _stop_webcam_preview(self):
        self.webcam_preview_running = False
        if hasattr(self, 'webcam_preview_cap') and self.webcam_preview_cap:
            try: self.webcam_preview_cap.release()
            except Exception: pass
            self.webcam_preview_cap = None
            self._preview_device_idx = None

    def _webcam_preview_tick(self):
        if not getattr(self, 'webcam_preview_running', False): return
        cap = getattr(self, 'webcam_preview_cap', None)
        if not cap: return
        ok, frame = cap.read()
        if ok:
            # Target fixed preview box (letterbox inside to avoid UI resizing)
            target_w = getattr(self, '_preview_w', cfg.get('preview_max_width', 480))
            target_h = getattr(self, '_preview_h', cfg.get('preview_max_height', 360))
            h,w = frame.shape[:2]
            if w == 0 or h == 0:
                return
            scale = min(target_w/float(w), target_h/float(h))
            new_w, new_h = max(1,int(w*scale)), max(1,int(h*scale))
            resized = cv2.resize(frame,(new_w,new_h))
            canvas = np.zeros((target_h,target_w,3), dtype=np.uint8)
            ox, oy = (target_w - new_w)//2, (target_h - new_h)//2
            canvas[oy:oy+new_h, ox:ox+new_w] = resized
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
            if hasattr(self,'webcam_preview_panel'):
                self.webcam_preview_panel.configure(image=tkimg)
                self.webcam_preview_panel.image = tkimg
        self.root.after(1000//max(1,cfg.get('camera_fps',30)), self._webcam_preview_tick)

    # -------------- Object Classification (ROI capture) --------------
    def open_object_classification(self):
        """Dialog A: Streaming preview for object classification.
        Opens a live webcam stream inside the dialog. Pressing Capture
        grabs the current frame and opens a new dialog displaying the
        still image with options to train master or a single object.
        """
        if self._object_classification_window and self._object_classification_window.winfo_exists():
            self._object_classification_window.lift(); return
        win = tk.Toplevel(self.root)
        win.title('Object Classification')
        self._object_classification_window = win
        self._obj_prev_w = cfg.get('preview_max_width', 480)
        self._obj_prev_h = cfg.get('preview_max_height', 360)
        blank = ImageTk.PhotoImage(Image.fromarray(np.zeros((self._obj_prev_h,self._obj_prev_w,3),dtype=np.uint8)))
        self.obj_preview_panel = tk.Label(win, image=blank, width=self._obj_prev_w, height=self._obj_prev_h, relief='sunken')
        self.obj_preview_panel.image = blank
        self.obj_preview_panel.pack(padx=6, pady=6)
        btn_row = ttk.Frame(win); btn_row.pack(pady=4)
        self._btn_capture = ttk.Button(btn_row, text='Capture', command=self._capture_obj_stream_frame)
        self._btn_capture.pack(side='left', padx=4)
        ttk.Button(btn_row, text='OK', command=lambda: self._finalize_object_classification(ok_close=True)).pack(side='left', padx=4)
        ttk.Button(btn_row, text='Close', command=lambda: _close()).pack(side='left', padx=4)
        self._last_capture_frame = None
        # Start streaming capture for this dialog
        self._start_object_stream()
        def _close():
            if win.winfo_exists():
                win.destroy()
            self._object_classification_window=None
            self._stop_object_stream()
        win.protocol('WM_DELETE_WINDOW', _close)

    def _finalize_object_classification(self, ok_close=False):
        """Called when user presses OK in Object Classification dialog.
        Immediately closes the dialog, opens a progress window and starts training.
        After training it reloads model weights automatically.
        """
        # Close dialog immediately if requested
        if ok_close and getattr(self,'_object_classification_window', None):
            try:
                if self._object_classification_window.winfo_exists():
                    self._object_classification_window.destroy()
            except Exception: pass
            self._object_classification_window = None
        # Stop stream
        self._stop_object_stream()
        data_dir = cfg.get('data_dir','data')
        images_dir = os.path.join(data_dir,'images')
        labels_dir = os.path.join(data_dir,'labels')
        if not os.path.isdir(images_dir) or not os.listdir(images_dir):
            messagebox.showwarning('Train','No images captured yet. Capture & label objects first.')
            return
        if not os.path.isdir(labels_dir) or not any(f.endswith('.txt') and f!='classes.json' for f in os.listdir(labels_dir)):
            messagebox.showwarning('Train','No label files found. Train highlighted objects first.')
            return
        # Open progress window
        self._open_training_progress_window()
        self.status_var.set('Training...')

        def _train_and_reload():
            ok = False
            try:
                import trainer
                self._log_training_msg('Starting training...')
                # Capture stdout/stderr
                q = self._train_log_queue
                class _Writer:
                    def write(self_inner, s):
                        if s.strip():
                            q.put(s.replace('\r','\n'))
                        return len(s)
                    def flush(self_inner):
                        pass
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = _Writer()
                try:
                    ok = trainer.train(data_dir=data_dir)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                if ok:
                    self._log_training_msg('Copying best weights...')
                    self._attempt_copy_best_weights()
                    self._log_training_msg('Reloading model...')
                    try:
                        self.model._attempt_load()
                    except Exception as e:
                        self._log_training_msg(f'Reload failed: {e}')
                    self.status_var.set('Model trained & loaded')
                    self._log_training_msg('Training complete.')
                else:
                    self.status_var.set('Ready')
                    self._log_training_msg('Training aborted or not started.')
            except Exception as e:
                self.status_var.set('Ready')
                self._log_training_msg(f'Error: {e}')
            finally:
                # finalize UI
                self._training_finish_ui(ok)
        threading.Thread(target=_train_and_reload, daemon=True).start()

    def _open_training_progress_window(self):
        if getattr(self,'_train_progress_win', None) and self._train_progress_win.winfo_exists():
            return
        win = tk.Toplevel(self.root)
        win.title('Training Progress')
        win.geometry('520x320')
        ttk.Label(win, text='Model Training in Progress').pack(anchor='w', padx=8, pady=(8,4))
        txt = tk.Text(win, width=70, height=14, wrap='word', state='disabled')
        txt.pack(fill='both', expand=True, padx=8, pady=4)
        pb = ttk.Progressbar(win, mode='indeterminate')
        pb.pack(fill='x', padx=8, pady=(0,6))
        pb.start(90)
        self._train_progress_win = win
        self._train_progress_text = txt
        self._train_progress_bar = pb
        self._train_log_queue = queue.Queue()
        self.root.after(150, self._drain_train_log_queue)
        win.protocol('WM_DELETE_WINDOW', lambda: None)  # disable manual close during training

    def _log_training_msg(self, msg):
        if not getattr(self,'_train_log_queue', None):
            return
        self._train_log_queue.put(msg + '\n')

    def _drain_train_log_queue(self):
        if not getattr(self,'_train_log_queue', None):
            return
        updated = False
        try:
            while True:
                line = self._train_log_queue.get_nowait()
                if getattr(self,'_train_progress_text', None):
                    self._train_progress_text.configure(state='normal')
                    self._train_progress_text.insert('end', line)
                    self._train_progress_text.see('end')
                    self._train_progress_text.configure(state='disabled')
                    updated = True
        except queue.Empty:
            pass
        if getattr(self,'_train_progress_win', None) and self._train_progress_win.winfo_exists():
            self.root.after(200, self._drain_train_log_queue)

    def _training_finish_ui(self, success):
        if getattr(self,'_train_progress_bar', None):
            try: self._train_progress_bar.stop()
            except Exception: pass
        if getattr(self,'_train_progress_text', None):
            self._train_progress_text.configure(state='normal')
            self._train_progress_text.insert('end', '\nDone.' if success else '\nFinished with issues.')
            self._train_progress_text.see('end')
            self._train_progress_text.configure(state='disabled')
        if getattr(self,'_train_progress_win', None) and self._train_progress_win.winfo_exists():
            # allow close now
            self._train_progress_win.protocol('WM_DELETE_WINDOW', self._train_progress_win.destroy)
            # add close button if not present
            if not hasattr(self,'_train_progress_close_added'):
                btn = ttk.Button(self._train_progress_win, text='Close', command=self._train_progress_win.destroy)
                btn.pack(pady=6)
                self._train_progress_close_added = True

    def _attempt_copy_best_weights(self):
        """Search latest Ultralytics run folders for best.pt and copy into models/best.pt."""
        try:
            runs_dir = os.path.join(os.getcwd(),'runs')
            if not os.path.isdir(runs_dir):
                return
            # Collect candidate best.pt paths (detect/train*/weights/best.pt)
            candidates = glob.glob(os.path.join(runs_dir,'detect','train*','weights','best.pt'))
            if not candidates:
                # generic pattern
                candidates = glob.glob(os.path.join(runs_dir,'**','best.pt'), recursive=True)
            if not candidates:
                return
            # Pick most recently modified
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            best_src = candidates[0]
            models_dir = cfg.get('models_dir','models')
            os.makedirs(models_dir, exist_ok=True)
            dst = os.path.join(models_dir,'best.pt')
            shutil.copy2(best_src, dst)
            print(f'[training] Copied best weights from {best_src} to {dst}')
        except Exception as e:
            print(f'[training] Could not copy best weights: {e}')

    # --- Object classification streaming helpers ---
    def _start_object_stream(self):
        if getattr(self, '_obj_stream_running', False):
            return
        cap = cv2.VideoCapture(self.device_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror('Error','Cannot open webcam for classification'); return
        self._obj_preview_cap = cap
        self._obj_stream_running = True
        self._object_stream_tick()

    def _stop_object_stream(self):
        self._obj_stream_running = False
        if hasattr(self,'_obj_preview_cap') and self._obj_preview_cap:
            try: self._obj_preview_cap.release()
            except Exception: pass
            self._obj_preview_cap = None

    def _object_stream_tick(self):
        if not getattr(self,'_obj_stream_running', False): return
        cap = getattr(self,'_obj_preview_cap', None)
        if not cap: return
        ok, frame = cap.read()
        if ok:
            target_w, target_h = self._obj_prev_w, self._obj_prev_h
            h,w = frame.shape[:2]
            if h>0 and w>0:
                scale = min(target_w/float(w), target_h/float(h))
                new_w,new_h = max(1,int(w*scale)), max(1,int(h*scale))
                resized = cv2.resize(frame,(new_w,new_h))
                canvas = np.zeros((target_h,target_w,3),dtype=np.uint8)
                ox,oy=(target_w-new_w)//2,(target_h-new_h)//2
                canvas[oy:oy+new_h, ox:ox+new_w]=resized
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
                if hasattr(self,'obj_preview_panel'):
                    self.obj_preview_panel.configure(image=tkimg)
                    self.obj_preview_panel.image = tkimg
        # schedule next
        if getattr(self,'_object_classification_window', None) and self._object_classification_window.winfo_exists():
            self.root.after(1000//max(1,cfg.get('camera_fps',30)), self._object_stream_tick)

    # --- Capture from stream and open dedicated dialog ---
    def _capture_obj_stream_frame(self):
        if not getattr(self,'_obj_preview_cap', None):
            messagebox.showinfo('Info','Stream not running'); return
        ok, frame = self._obj_preview_cap.read()
        if not ok or frame is None:
            messagebox.showerror('Error','Failed to capture frame'); return
        self._last_capture_frame = frame.copy()
        self._open_captured_image_dialog()

    def _open_captured_image_dialog(self):
        if self._last_capture_frame is None:
            return
        if hasattr(self,'_captured_image_win') and self._captured_image_win and self._captured_image_win.winfo_exists():
            self._captured_image_win.lift(); return
        win = tk.Toplevel(self.root)
        win.title('Captured Image')
        self._captured_image_win = win
        container = ttk.Frame(win)
        container.pack(fill='both', expand=True, padx=6, pady=6)
        # Left: image display area
        left = ttk.Frame(container)
        left.pack(side='left', fill='both', expand=True)
        self._captured_canvas_holder = left
        self.captured_image_canvas = tk.Canvas(left, relief='sunken', cursor='tcross')
        self.captured_image_canvas.pack(fill='both', expand=True)
        # Right: trained data list + action buttons
        right = ttk.Frame(container)
        right.pack(side='left', fill='y', padx=(8,0))
        ttk.Label(right, text='Trained Data').pack(anchor='w')
        self.trained_listbox = tk.Listbox(right, width=32, height=18)
        self.trained_listbox.pack(fill='y', expand=False, pady=(2,4))
        self.trained_listbox.bind('<<ListboxSelect>>', self._on_trained_item_select)
        actions = ttk.Frame(right)
        actions.pack(anchor='w', pady=(4,0))
        self._btn_edit_trained = ttk.Button(actions, text='Edit', width=6, command=self._edit_trained_item, state='disabled')
        self._btn_edit_trained.pack(side='left', padx=(0,4))
        self._btn_delete_trained = ttk.Button(actions, text='Delete', width=8, command=self._delete_trained_item, state='disabled')
        self._btn_delete_trained.pack(side='left')
        # Mouse interaction bindings
        self.captured_image_canvas.bind('<ButtonPress-1>', self._captured_image_press)
        self.captured_image_canvas.bind('<B1-Motion>', self._captured_image_motion)
        self.captured_image_canvas.bind('<ButtonRelease-1>', self._captured_image_release)
        # Progress bar
        self.captured_progress = ttk.Progressbar(right, mode='indeterminate', length=160)
        self.captured_progress.pack(pady=4)
        self.captured_progress.pack_forget()
        # Close handler
        def _close():
            if win.winfo_exists():
                win.destroy()
            self._captured_image_win = None
        # Bottom buttons
        btns = ttk.Frame(win)
        btns.pack(pady=4)
        ttk.Button(btns, text='Train Master', command=self._quick_save_master_from_capture).pack(side='left', padx=4)
        self._btn_train_highlighted = ttk.Button(btns, text='Train Highlighted Object', command=self._open_train_object_from_capture, state='disabled')
        self._btn_train_highlighted.pack(side='left', padx=4)
        ttk.Button(btns, text='Close', command=_close).pack(side='left', padx=4)
        # Populate initial image
        self._display_captured_frame(self._last_capture_frame)
        try:
            self._load_trained_dataset_items()
        except Exception:
            pass
        win.protocol('WM_DELETE_WINDOW', _close)

    def _display_captured_frame(self, frame):
        if frame is None: return
        # Store original full-resolution frame
        self._captured_full_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w = frame.shape[:2]
        max_dim = 1000
        scale = min(1.0, max_dim/float(max(h,w)))
        self._captured_disp_scale = scale
        if scale < 1.0:
            disp = cv2.resize(rgb, (int(w*scale), int(h*scale)))
        else:
            disp = rgb
        self._captured_disp_image = disp  # numpy RGB for later overlay
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self._captured_photo = photo
        if hasattr(self,'captured_image_canvas'):
            self.captured_image_canvas.delete('all')
            self.captured_image_canvas.config(width=disp.shape[1], height=disp.shape[0])
            self.captured_image_canvas.create_image(0,0, anchor='nw', image=photo, tags='base')

    def _load_trained_dataset_items(self):
        data_dir = cfg.get('data_dir','data')
        img_dir = os.path.join(data_dir,'images')
        lbl_dir = os.path.join(data_dir,'labels')
        items = []
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                    continue
                stem, _ = os.path.splitext(fname)
                lbl_path = os.path.join(lbl_dir, stem + '.txt')
                if os.path.exists(lbl_path):
                    items.append((stem, os.path.join(img_dir, fname), lbl_path))
        self._trained_items = items
        if hasattr(self,'trained_listbox'):
            self.trained_listbox.delete(0, tk.END)
            for stem, _, _ in items:
                self.trained_listbox.insert(tk.END, stem)
            if not items:
                self.trained_listbox.insert(tk.END, '(none)')

    def _on_trained_item_select(self, event):
        if not hasattr(self, 'trained_listbox'):
            return
        sel = self.trained_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if not hasattr(self, '_trained_items') or idx >= len(self._trained_items):
            return
        if hasattr(self, '_btn_edit_trained'):
            try:
                self._btn_edit_trained.configure(state='normal')
                self._btn_delete_trained.configure(state='normal')
            except Exception:
                pass
        # Do not modify displayed captured image (requirement)
        self._selected_trained_index = idx

    def _edit_trained_item(self):
        if not hasattr(self,'trained_listbox'): return
        sel = self.trained_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if not hasattr(self,'_trained_items') or idx >= len(self._trained_items): return
        stem, img_path, lbl_path = self._trained_items[idx]
        # parse first bbox from label for crop preview
        crop_img = None
        try:
            frame = cv2.imread(img_path)
            if frame is not None and os.path.exists(lbl_path):
                with open(lbl_path,'r',encoding='utf-8') as f:
                    line = f.readline().strip().split()
                    if len(line)==5:
                        _, cx, cy, bw, bh = line
                        h,w = frame.shape[:2]
                        cx=float(cx); cy=float(cy); bw=float(bw); bh=float(bh)
                        x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h); x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
                        x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
                        crop_img = frame[y1:y2, x1:x2]
        except Exception:
            crop_img = None
        dlg = tk.Toplevel(self.root); dlg.title('Edit Trained Object')
        dlg.transient(self.root)
        if crop_img is not None and crop_img.size>0:
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            max_w=240
            scale = min(1.0, max_w/float(rgb.shape[1]))
            if scale<1.0:
                rgb = cv2.resize(rgb,(int(rgb.shape[1]*scale), int(rgb.shape[0]*scale)))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl_preview = tk.Label(dlg, image=photo); lbl_preview.image=photo; lbl_preview.pack(padx=8,pady=(8,4))
        ttk.Label(dlg, text='Object Name:').pack(anchor='w', padx=8)
        name_var = tk.StringVar(value=stem)
        entry = ttk.Entry(dlg, textvariable=name_var); entry.pack(fill='x', padx=8, pady=(0,8)); entry.focus_set()
        def _ok():
            new_name = name_var.get().strip()
            if not new_name or new_name == stem:
                dlg.destroy(); return
            data_dir = cfg.get('data_dir','data')
            img_dir = os.path.join(data_dir,'images')
            lbl_dir = os.path.join(data_dir,'labels')
            new_img = os.path.join(img_dir, new_name + os.path.splitext(img_path)[1])
            new_lbl = os.path.join(lbl_dir, new_name + '.txt')
            try:
                os.rename(img_path, new_img)
                os.rename(lbl_path, new_lbl)
            except Exception as e:
                messagebox.showerror('Error', f'Could not rename: {e}')
                return
            dlg.destroy()
            self._load_trained_dataset_items()
            for i,(s,ip,lp) in enumerate(self._trained_items):
                if s == new_name:
                    self.trained_listbox.selection_clear(0, tk.END)
                    self.trained_listbox.selection_set(i)
                    self.trained_listbox.event_generate('<<ListboxSelect>>')
                    break
        btn_row = ttk.Frame(dlg); btn_row.pack(pady=4)
        ttk.Button(btn_row, text='OK', command=_ok).pack(side='left', padx=4)
        ttk.Button(btn_row, text='Cancel', command=dlg.destroy).pack(side='left', padx=4)
        dlg.bind('<Return>', lambda e: _ok())
        dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)

    def _delete_trained_item(self):
        if not hasattr(self,'trained_listbox'): return
        sel = self.trained_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if not hasattr(self,'_trained_items') or idx >= len(self._trained_items): return
        stem, img_path, lbl_path = self._trained_items[idx]
        if not messagebox.askyesno('Delete', f'Delete trained item "{stem}"?'):
            return
        try:
            if os.path.exists(img_path): os.remove(img_path)
            if os.path.exists(lbl_path): os.remove(lbl_path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not delete: {e}')
            return
        self._load_trained_dataset_items()
        if hasattr(self,'_btn_edit_trained'):
            try:
                self._btn_edit_trained.configure(state='disabled')
                self._btn_delete_trained.configure(state='disabled')
            except Exception:
                pass

    # --- Click-based auto-detect & highlight on captured image ---
    # (GrabCut removed) legacy single-click disabled

    # Drag-based rectangle selection handlers
    def _captured_image_press(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_rect_id = None
    def _captured_image_motion(self, event):
        if not hasattr(self,'_drag_start') or self._drag_start is None:
            return
        sx, sy = self._drag_start
        ex, ey = event.x, event.y
        if hasattr(self,'captured_image_canvas'):
            c = self.captured_image_canvas
            if self._drag_rect_id is not None:
                c.coords(self._drag_rect_id, sx, sy, ex, ey)
            else:
                self._drag_rect_id = c.create_rectangle(sx, sy, ex, ey, outline='cyan', width=2, dash=(4,2), tags='drag_rect')
    def _captured_image_release(self, event):
        if not hasattr(self,'_drag_start') or self._drag_start is None:
            return
        sx, sy = self._drag_start; ex, ey = event.x, event.y
        self._drag_start = None
        # Require a meaningful rectangle; ignore if too small
        if abs(ex-sx) < 10 or abs(ey-sy) < 10:
            return
        # Normalize rect
        x1, x2 = sorted([sx, ex])
        y1, y2 = sorted([sy, ey])
        scale = getattr(self,'_captured_disp_scale', 1.0) or 1.0
        # Map to original image coordinates
        ox1, oy1 = int(x1/scale), int(y1/scale)
        ox2, oy2 = int(x2/scale), int(y2/scale)
        if ox2-ox1 < 10 or oy2-oy1 < 10:
            return
        # Store bbox
        self._last_highlight_bbox = (ox1, oy1, ox2, oy2)
        # Build highlighted display image directly (no GrabCut)
        if hasattr(self,'_captured_full_frame') and self._captured_full_frame is not None:
            frame = self._captured_full_frame.copy()
            base_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = base_rgb.copy()
            highlight_alpha = cfg.get('highlight_alpha', 0.4)
            overlay[oy1:oy2, ox1:ox2] = (255,0,0)
            blended = base_rgb.copy()
            cv2.addWeighted(overlay, highlight_alpha, blended, 1-highlight_alpha, 0, blended)
            cv2.rectangle(blended,(ox1,oy1),(ox2,oy2),(255,255,255),2)
            if scale < 1.0:
                disp = cv2.resize(blended, (int(base_rgb.shape[1]*scale), int(base_rgb.shape[0]*scale)))
            else:
                disp = blended
            # update canvas
            if hasattr(self,'captured_image_canvas'):
                self._captured_disp_image = disp
                photo = ImageTk.PhotoImage(Image.fromarray(disp))
                self._captured_photo = photo
                self.captured_image_canvas.delete('all')
                self.captured_image_canvas.config(width=disp.shape[1], height=disp.shape[0])
                self.captured_image_canvas.create_image(0,0, anchor='nw', image=photo, tags='base')
            if hasattr(self,'_btn_train_highlighted'):
                try: self._btn_train_highlighted.configure(state='normal')
                except Exception: pass
        # done

    def _open_train_object_from_capture(self):
        # New behavior: use existing highlighted bbox from GrabCut
        if self._last_capture_frame is None:
            messagebox.showinfo('Info','Capture an image first'); return
        if not hasattr(self,'_last_highlight_bbox'):
            messagebox.showinfo('Info','Highlight an object first'); return
        x1,y1,x2,y2 = self._last_highlight_bbox
        if x2<=x1 or y2<=y1:
            messagebox.showerror('Error','Invalid highlighted region'); return
        crop = self._last_capture_frame[y1:y2, x1:x2]
        if crop is None or crop.size==0:
            messagebox.showerror('Error','Could not crop highlighted region'); return
        win = tk.Toplevel(self.root); win.title('Train Highlighted Object')
        win.transient(self.root)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        disp_w = 220
        scale = disp_w / float(rgb.shape[1])
        disp_h = int(rgb.shape[0]*scale)
        disp = cv2.resize(rgb, (disp_w, disp_h)) if scale != 1.0 else rgb
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        img_label = tk.Label(win, image=photo); img_label.image = photo; img_label.pack(padx=8,pady=(8,4))
        ttk.Label(win, text='Object Name:').pack(anchor='w', padx=8)
        name_var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=name_var)
        entry.pack(fill='x', padx=8, pady=(0,8))
        entry.focus_set()
        def _ok():
            name = name_var.get().strip() or f"obj_{int(time.time())}"
            # classes management
            if not hasattr(self,'_classes'):
                self._classes_path = os.path.join(cfg.get('data_dir','data'),'labels','classes.json')
                self._classes = self._load_classes()
            if name not in self._classes:
                self._classes.append(name); self._save_classes()
            cls_idx = self._classes.index(name)
            saved_name = self._persist_highlight_object(name, cls_idx, (x1,y1,x2,y2))
            win.destroy()
            self._load_trained_dataset_items()
            messagebox.showinfo('Saved', f'Object "{saved_name}" saved.')
        btn_row = ttk.Frame(win); btn_row.pack(pady=4)
        ttk.Button(btn_row, text='OK', command=_ok).pack(side='left', padx=4)
        ttk.Button(btn_row, text='Cancel', command=win.destroy).pack(side='left', padx=4)
        win.bind('<Return>', lambda e: _ok())
        win.protocol('WM_DELETE_WINDOW', win.destroy)

    def _captured_train_object_click(self, event):
        if self._last_capture_frame is None:
            return
        frame = self._last_capture_frame
        h,w = frame.shape[:2]
        x,y = int(event.x), int(event.y)
        half=100
        x0,y0=max(0,x-half), max(0,y-half)
        x1,y1=min(w,x+half), min(h,y+half)
        patch = frame[y0:y1, x0:x1]
        if patch.size==0: return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)
        contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c=max(contours,key=cv2.contourArea)
            if cv2.contourArea(c)>15:
                bx,by,bw,bh = cv2.boundingRect(c)
                rx1,ry1,rx2,ry2 = x0+bx, y0+by, x0+bx+bw, y0+by+bh
            else:
                rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        else:
            rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        # draw temp box
        if hasattr(self,'captured_single_canvas'):
            self.captured_single_canvas.create_rectangle(rx1,ry1,rx2,ry2, outline='yellow', width=2, tags='temp_box')
        self._open_single_object_name_dialog_capture([rx1,ry1,rx2,ry2])

    def _open_single_object_name_dialog_capture(self, bbox):
        win = tk.Toplevel(self.root); win.title('Name Object (Captured)')
        x1,y1,x2,y2 = bbox
        crop = self._last_capture_frame[y1:y2, x1:x2]
        if crop.size>0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((160,120)))
            lbl = tk.Label(win, image=photo); lbl.image=photo; lbl.pack(padx=6,pady=4)
        ttk.Label(win, text='Object name:').pack(anchor='w', padx=6)
        name_var = tk.StringVar(); entry = ttk.Entry(win, textvariable=name_var); entry.pack(fill='x', padx=6, pady=4); entry.focus_set()
        def _ok():
            name = name_var.get().strip() or f"obj_{int(time.time())}"
            if not hasattr(self,'_classes'):
                self._classes_path = os.path.join(cfg.get('data_dir','data'),'labels','classes.json')
                self._classes = self._load_classes()
            if name not in self._classes:
                self._classes.append(name); self._save_classes()
            cls_idx = self._classes.index(name)
            saved_name = self._persist_single_object_captured(name, cls_idx, bbox)
            win.destroy(); messagebox.showinfo('Saved', f'Object "{saved_name}" saved.')
        ttk.Button(win, text='OK', command=_ok).pack(pady=4)
        ttk.Button(win, text='Cancel', command=win.destroy).pack()
        win.transient(self.root); win.grab_set()

    def _persist_single_object_captured(self, name, cls_idx, bbox):
        data_dir = cfg.get('data_dir','data')
        img_dir = os.path.join(data_dir,'images'); os.makedirs(img_dir, exist_ok=True)
        lbl_dir = os.path.join(data_dir,'labels'); os.makedirs(lbl_dir, exist_ok=True)
        base_name = name
        counter = 1
        img_path = os.path.join(img_dir, f'{base_name}.jpg')
        lbl_path = os.path.join(lbl_dir, f'{base_name}.txt')
        while os.path.exists(img_path) or os.path.exists(lbl_path):
            candidate = f"{base_name}_{counter}"
            img_path = os.path.join(img_dir, f'{candidate}.jpg')
            lbl_path = os.path.join(lbl_dir, f'{candidate}.txt')
            counter += 1
        # derive final (possibly suffixed) name for label content association
        final_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(img_path, self._last_capture_frame)
        h,w = self._last_capture_frame.shape[:2]
        x1,y1,x2,y2 = bbox
        cx=(x1+x2)/2.0 / w; cy=(y1+y2)/2.0 / h; bw=(x2-x1)/w; bh=(y2-y1)/h
        with open(lbl_path,'w',encoding='utf-8') as f:
            f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return final_name

    def _persist_highlight_object(self, name, cls_idx, bbox):
        # identical to single object capture but uses provided bbox
        return self._persist_single_object_captured(name, cls_idx, bbox)


    def _capture_master_frame(self):
        """Capture a single frame for master / object training.
        Reuses an existing running capture (detector preview) if available to avoid
        Windows driver conflicts with simultaneous opens. Falls back to a fresh
        one-shot VideoCapture otherwise.
        """
        frame = None
        # Try reusing active detector / pipeline camera first (avoids second handle)
        if hasattr(self, 'detector') and getattr(self, 'detector', None) and getattr(self.detector, 'cap', None):
            try:
                ok_live, live_frame = self.detector.cap.read()
                if ok_live and live_frame is not None:
                    frame = live_frame
            except Exception:
                frame = None
        # If no frame yet, open temporary capture
        if frame is None:
            cap = cv2.VideoCapture(self.device_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                messagebox.showerror('Error','Cannot open webcam (in use?)'); return
            ok, tmp = cap.read(); cap.release()
            if not ok or tmp is None:
                messagebox.showerror('Error','Failed to capture frame'); return
            frame = tmp

        self._last_capture_frame = frame.copy()
        max_w,max_h = self._obj_prev_w,self._obj_prev_h
        h,w = frame.shape[:2]
        scale = min(max_w/float(w), max_h/float(h))
        new_w,new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(frame,(new_w,new_h))
        canvas = np.zeros((max_h,max_w,3),dtype=np.uint8)
        ox,oy=(max_w-new_w)//2,(max_h-new_h)//2
        canvas[oy:oy+new_h, ox:ox+new_w]=resized
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.obj_preview_panel.configure(image=tkimg); self.obj_preview_panel.image=tkimg
        self._btn_train_master.configure(state='normal')
        messagebox.showinfo('Captured','Image captured. Train Master or Train Object now.')

    # -------- Dialog B: Master Trainer --------
    def _open_master_trainer(self):
        if self._last_capture_frame is None:
            messagebox.showinfo('Info','Capture a frame first'); return
        if hasattr(self,'_master_trainer_win') and self._master_trainer_win and self._master_trainer_win.winfo_exists():
            self._master_trainer_win.lift(); return
        win = tk.Toplevel(self.root); win.title('Master Trainer')
        self._master_trainer_win = win
        self._annotations = []  # list of {name,class,bbox:[x1,y1,x2,y2]}
        # load classes list
        self._classes_path = os.path.join(cfg.get('data_dir','data'),'labels','classes.json')
        self._classes = self._load_classes()
        # image display
        self._master_image = self._last_capture_frame.copy()
        rgb = cv2.cvtColor(self._master_image, cv2.COLOR_BGR2RGB)
        self._master_im_pil = Image.fromarray(rgb)
        self._master_photo = ImageTk.PhotoImage(self._master_im_pil)
        self.master_canvas = tk.Canvas(win, width=self._master_photo.width(), height=self._master_photo.height(), cursor='tcross')
        self.master_canvas.pack(side='left', padx=6, pady=6)
        self.master_canvas.create_image(0,0, anchor='nw', image=self._master_photo)
        self.master_canvas.bind('<Button-1>', self._master_click)
        # side panel
        side = ttk.Frame(win); side.pack(side='left', fill='y', padx=6, pady=6)
        self.ann_list = tk.Listbox(side, width=32, height=18); self.ann_list.pack(fill='y')
        btns = ttk.Frame(side); btns.pack(pady=4)
        ttk.Button(btns, text='Save Master', command=self._save_master).grid(row=0,column=0,padx=2,pady=2)
        ttk.Button(btns, text='Clear Annotations', command=self._clear_annotations).grid(row=0,column=1,padx=2,pady=2)
        ttk.Button(btns, text='Close', command=lambda: self._close_master_trainer()).grid(row=0,column=2,padx=2,pady=2)
        win.protocol('WM_DELETE_WINDOW', self._close_master_trainer)

    def _close_master_trainer(self):
        if hasattr(self,'_master_trainer_win') and self._master_trainer_win and self._master_trainer_win.winfo_exists():
            self._master_trainer_win.destroy()
        self._master_trainer_win=None

    def _load_classes(self):
        path = os.path.join(cfg.get('data_dir','data'),'labels')
        os.makedirs(path, exist_ok=True)
        cpath = os.path.join(path,'classes.json')
        if os.path.exists(cpath):
            try:
                with open(cpath,'r',encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_classes(self):
        path = os.path.join(cfg.get('data_dir','data'),'labels')
        os.makedirs(path, exist_ok=True)
        cpath = os.path.join(path,'classes.json')
        try:
            with open(cpath,'w',encoding='utf-8') as f:
                json.dump(self._classes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('[classes] save failed', e)

    def _master_click(self, event):
        # simple fixed-size local region + edge refinement
        x, y = int(event.x), int(event.y)
        h, w = self._master_image.shape[:2]
        half=80
        x0,y0=max(0,x-half), max(0,y-half)
        x1,y1=min(w,x+half), min(h,y+half)
        patch = self._master_image[y0:y1, x0:x1]
        if patch.size==0: return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)
        contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c=max(contours,key=cv2.contourArea)
            if cv2.contourArea(c)>15:
                bx,by,bw,bh = cv2.boundingRect(c)
                rx1,ry1,rx2,ry2 = x0+bx, y0+by, x0+bx+bw, y0+by+bh
            else:
                rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        else:
            rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        self._open_name_dialog([rx1,ry1,rx2,ry2])

    def _open_name_dialog(self, bbox):
        win = tk.Toplevel(self.root); win.title('Name Object')
        # cropped preview
        x1,y1,x2,y2 = bbox
        crop = self._master_image[y1:y2, x1:x2]
        if crop.size>0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((160,120)))
            lbl = tk.Label(win, image=photo); lbl.image=photo; lbl.pack(padx=6,pady=4)
        ttk.Label(win, text='Object name:').pack(anchor='w', padx=6)
        name_var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=name_var); entry.pack(fill='x', padx=6, pady=4); entry.focus_set()
        def _ok():
            name = name_var.get().strip() or f"obj_{len(self._annotations)}"
            if name not in self._classes:
                self._classes.append(name)
                self._save_classes()
            cls_idx = self._classes.index(name)
            self._annotations.append({'name':name,'class':cls_idx,'bbox':bbox})
            self._refresh_annotations()
            win.destroy()
        ttk.Button(win, text='OK', command=_ok).pack(pady=4)
        ttk.Button(win, text='Cancel', command=win.destroy).pack()
        win.transient(self._master_trainer_win)
        win.grab_set()

    def _refresh_annotations(self):
        if not hasattr(self,'ann_list'): return
        self.ann_list.delete(0, tk.END)
        for ann in self._annotations:
            x1,y1,x2,y2 = ann['bbox']
            self.ann_list.insert(tk.END, f"{ann['name']} | cls {ann['class']} | {x2-x1}x{y2-y1}")
        # redraw boxes
        if hasattr(self,'master_canvas'):
            self.master_canvas.delete('box')
            for ann in self._annotations:
                x1,y1,x2,y2 = ann['bbox']
                self.master_canvas.create_rectangle(x1,y1,x2,y2, outline='lime', width=2, tags='box')

    def _clear_annotations(self):
        if not getattr(self,'_annotations', None): return
        if not messagebox.askyesno('Confirm','Clear all annotations?'): return
        self._annotations.clear(); self._refresh_annotations()

    def _save_master(self):
        if not hasattr(self,'_annotations'):
            messagebox.showerror('Error','No annotations'); return
        if not self._annotations:
            if not messagebox.askyesno('Empty','Save master with zero annotations?'):
                return
        master_dir = cfg.get('master_dir','data/master')
        os.makedirs(master_dir, exist_ok=True)
        img_path = os.path.join(master_dir, 'master.jpg')
        cv2.imwrite(img_path, self._master_image)
        # labels
        h,w = self._master_image.shape[:2]
        lines=[]
        for ann in self._annotations:
            x1,y1,x2,y2 = ann['bbox']
            cx = (x1+x2)/2.0 / w
            cy = (y1+y2)/2.0 / h
            bw = (x2-x1)/w
            bh = (y2-y1)/h
            lines.append(f"{ann['class']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(master_dir,'master_labels.txt'),'w',encoding='utf-8') as f:
            f.write('\n'.join(lines))
        cfg['last_master_save'] = int(time.time())
        save_config(cfg)
        messagebox.showinfo('Saved','Master image & labels saved.')

    def _quick_save_master_from_capture(self):
        if self._last_capture_frame is None:
            messagebox.showinfo('Info','No captured frame to save as master'); return
        master_dir = cfg.get('master_dir','data/master')
        os.makedirs(master_dir, exist_ok=True)
        img_path = os.path.join(master_dir, 'master.jpg')
        cv2.imwrite(img_path, self._last_capture_frame)
        cfg['last_master_save'] = int(time.time())
        save_config(cfg)
        messagebox.showinfo('Saved','Master image saved.')

    # --- (Old methods _start_obj_stream etc. removed in new workflow) ---

    # -------- Single Object Training (Dialog D) --------
    def _open_train_object_dialog(self):
        if not getattr(self,'_obj_preview_cap', None):
            messagebox.showinfo('Info','Open Object Classification preview first'); return
        if hasattr(self,'_train_object_win') and self._train_object_win and self._train_object_win.winfo_exists():
            self._train_object_win.lift(); return
        win = tk.Toplevel(self.root); win.title('Train Object')
        self._train_object_win = win
        # snapshot current frame
        ok, frame = self._obj_preview_cap.read()
        if not ok:
            messagebox.showerror('Error','Could not read frame'); win.destroy(); self._train_object_win=None; return
        self._single_object_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._single_obj_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.single_canvas = tk.Canvas(win, width=self._single_obj_photo.width(), height=self._single_obj_photo.height(), cursor='tcross')
        self.single_canvas.pack(padx=6,pady=6)
        self.single_canvas.create_image(0,0, anchor='nw', image=self._single_obj_photo)
        ttk.Label(win, text='Click the object to auto-detect its box.').pack(pady=(0,4))
        self.single_canvas.bind('<Button-1>', self._train_object_click)
        ttk.Button(win, text='Close', command=lambda: self._close_train_object_dialog()).pack(pady=4)
        win.protocol('WM_DELETE_WINDOW', self._close_train_object_dialog)

    def _close_train_object_dialog(self):
        if getattr(self,'_train_object_win', None) and self._train_object_win.winfo_exists():
            try: self._train_object_win.destroy()
            except Exception: pass
        self._train_object_win=None

    def _train_object_click(self, event):
        if not hasattr(self,'_single_object_frame'): return
        frame = self._single_object_frame
        h,w = frame.shape[:2]
        x,y = int(event.x), int(event.y)
        half=100
        x0,y0=max(0,x-half), max(0,y-half)
        x1,y1=min(w,x+half), min(h,y+half)
        patch = frame[y0:y1, x0:x1]
        if patch.size==0: return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)
        contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c=max(contours,key=cv2.contourArea)
            if cv2.contourArea(c)>15:
                bx,by,bw,bh = cv2.boundingRect(c)
                rx1,ry1,rx2,ry2 = x0+bx, y0+by, x0+bx+bw, y0+by+bh
            else:
                rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        else:
            rx1,ry1,rx2,ry2 = x0,y0,x1,y1
        self.single_canvas.create_rectangle(rx1,ry1,rx2,ry2, outline='yellow', width=2, tags='temp_box')
        self._open_single_object_name_dialog([rx1,ry1,rx2,ry2])

    def _open_single_object_name_dialog(self, bbox):
        win = tk.Toplevel(self.root); win.title('Name Object')
        x1,y1,x2,y2 = bbox
        crop = self._single_object_frame[y1:y2, x1:x2]
        if crop.size>0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((160,120)))
            lbl = tk.Label(win, image=photo); lbl.image=photo; lbl.pack(padx=6,pady=4)
        ttk.Label(win, text='Object name:').pack(anchor='w', padx=6)
        name_var = tk.StringVar(); entry = ttk.Entry(win, textvariable=name_var); entry.pack(fill='x', padx=6, pady=4); entry.focus_set()
        def _ok():
            name = name_var.get().strip() or f"obj_{int(time.time())}"
            # load classes reuse
            if not hasattr(self,'_classes'):
                self._classes_path = os.path.join(cfg.get('data_dir','data'),'labels','classes.json')
                self._classes = self._load_classes()
            if name not in self._classes:
                self._classes.append(name)
                self._save_classes()
            cls_idx = self._classes.index(name)
            self._persist_single_object(name, cls_idx, bbox)
            win.destroy()
            messagebox.showinfo('Saved', f'Object "{name}" saved.')
        ttk.Button(win, text='OK', command=_ok).pack(pady=4)
        ttk.Button(win, text='Cancel', command=win.destroy).pack()
        win.transient(self._train_object_win)
        win.grab_set()

    def _persist_single_object(self, name, cls_idx, bbox):
        # Save full frame + label pair (one-object dataset item)
        data_dir = cfg.get('data_dir','data')
        img_dir = os.path.join(data_dir,'images'); os.makedirs(img_dir, exist_ok=True)
        lbl_dir = os.path.join(data_dir,'labels'); os.makedirs(lbl_dir, exist_ok=True)
        ts = int(time.time()*1000)
        img_path = os.path.join(img_dir, f'{name}_{ts}.jpg')
        cv2.imwrite(img_path, self._single_object_frame)
        h,w = self._single_object_frame.shape[:2]
        x1,y1,x2,y2 = bbox
        cx=(x1+x2)/2.0 / w; cy=(y1+y2)/2.0 / h; bw=(x2-x1)/w; bh=(y2-y1)/h
        with open(os.path.join(lbl_dir, f'{name}_{ts}.txt'),'w',encoding='utf-8') as f:
            f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # -------------- ChatBot Role --------------
    def open_chatbot_settings(self):
        if self._chatbot_settings_window and self._chatbot_settings_window.winfo_exists():
            self._chatbot_settings_window.lift(); return
        win = tk.Toplevel(self.root)
        win.title('ChatBot Settings')
        self._chatbot_settings_window = win
        # Gemini API Key input (placed above role)
        ttk.Label(win, text='Gemini API Key:').pack(anchor='w', padx=4, pady=(6,2))
        self.gemini_api_key_var = tk.StringVar(value=get_gemini_api_key(cfg))
        api_frame = ttk.Frame(win)
        api_frame.pack(fill='x', padx=6)
        self.gemini_api_key_entry = ttk.Entry(api_frame, textvariable=self.gemini_api_key_var, width=60, show='*')
        self.gemini_api_key_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(api_frame, text='Show', command=self._toggle_gemini_api_visibility).pack(side='left', padx=2)
        ttk.Button(api_frame, text='Clear', command=lambda: self.gemini_api_key_var.set('')).pack(side='left', padx=2)
        ttk.Label(win, text='ChatBot Role:').pack(anchor='w', padx=4, pady=(10,4))
        default_role = (
            "You are a friendly visual helper who compares a master (reference) image with a captured image from a camera.\n"
            "Your job is to:\n"
            "1) Look at the trained objects in the master image and the captured image.\n"
            "2) Identify differences such as:\n"
            " - Missing objects (present in master but not in captured).\n"
            " - Moved objects (location is different).\n"
            " - Replaced objects (different object at the same spot).\n"
            " - Resized objects (smaller or larger compared to master).\n"
            "3) Explain these differences in clear, simple language that both primary and secondary students can understand."
        )
        self.chatbot_role_text = tk.Text(win, width=70, height=12, wrap='word')
        self.chatbot_role_text.pack(fill='both', expand=True, padx=6, pady=6)
        self.chatbot_role_text.insert('1.0', cfg.get('chatbot_role', default_role))
        ttk.Button(win, text='Save', command=lambda: self._save_chatbot_role(default_role)).pack(pady=4)
        def _close():
            if win.winfo_exists(): win.destroy()
            self._chatbot_settings_window = None
        win.protocol('WM_DELETE_WINDOW', _close)

    def _save_chatbot_role(self, default_role):
        role = self.chatbot_role_text.get('1.0','end').strip() or default_role
        cfg['chatbot_role'] = role
        # Save Gemini API key (store even if blank so user can remove it intentionally)
        if hasattr(self, 'gemini_api_key_var'):
            set_gemini_api_key(cfg, self.gemini_api_key_var.get().strip())
        save_config(cfg)
        messagebox.showinfo('ChatBot','Settings saved to config.json')

    def _toggle_gemini_api_visibility(self):
        # Toggle between masked and plain text display of API key
        if not hasattr(self, 'gemini_api_key_entry'): return
        cur = self.gemini_api_key_entry.cget('show')
        self.gemini_api_key_entry.config(show='' if cur == '*' else '*')

    # -------------- Detection main loop --------------
    def start(self):
        # Start detection via engine if not already running
        # Start via detection engine
        if not hasattr(self, 'detector'):
            self.detector = DetectionEngine(
                self.root,
                self.model,
                cfg,
                match_fn=match_detections_to_master if callable(match_detections_to_master) else None,
                on_update=self._on_detection_update
            )
        if self.detector.is_running():
            return
        self.detector.start(self.device_idx, lambda: self.master)
        if not self.detector.is_running():
            messagebox.showerror('Error','Cannot open webcam');
            self.status_var.set(t('status_ready','Ready'))
            return
        self.status_var.set(t('status_detecting','Streaming...'))

    def stop(self):
        if hasattr(self,'detector'):
            self.detector.stop()
        self.status_var.set(t('status_ready','Ready'))

    def capture_image(self):
        if not hasattr(self,'detector') or not self.detector or not self.detector.cap:
            messagebox.showinfo('Info','Start stream first'); return
        ok, frame = self.detector.cap.read()
        if not ok:
            messagebox.showerror('Error','Failed to capture'); return
        out_dir = os.path.join(cfg.get('data_dir','data'),'images')
        os.makedirs(out_dir, exist_ok=True)
        fname = f"capture_{int(time.time())}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, frame)
        messagebox.showinfo('Saved', f'Saved {path}')

    def open_annotator(self):
        top = tk.Toplevel(self.root)
        Annotator(top)

    def train_model(self):
        data_dir = cfg.get('data_dir','data')
        images_dir = os.path.join(data_dir,'images')
        labels_dir = os.path.join(data_dir,'labels')
        if not os.path.isdir(images_dir) or not os.listdir(images_dir):
            messagebox.showwarning('Train','No images found. Capture/train objects first.')
            return
        if not os.path.isdir(labels_dir) or not any(f.endswith('.txt') for f in os.listdir(labels_dir)):
            messagebox.showwarning('Train','No label files found. Train highlighted objects first.')
            return
        self.status_var.set(t('status_training','Training...'))
        def _run():
            try:
                import trainer
                ok = trainer.train(data_dir=data_dir)
                if ok:
                    messagebox.showinfo('Training','Training complete (see console for details).')
                else:
                    messagebox.showwarning('Training','Training not started or aborted (see console).')
            except Exception as e:
                messagebox.showerror('Training', f'Failed: {e}')
            finally:
                self.status_var.set(t('status_ready','Ready'))
        threading.Thread(target=_run, daemon=True).start()

    def load_master(self):
        mdir = cfg.get('master_dir','data/master')
        os.makedirs(mdir, exist_ok=True)
        img_path = filedialog.askopenfilename(title='Select master image', filetypes=[('Images','*.jpg *.png *.jpeg')], initialdir=mdir)
        if not img_path: return
        lbl_path = filedialog.askopenfilename(title='Select master labels (YOLO)', filetypes=[('Text','*.txt')], initialdir=mdir)
        if not lbl_path:
            messagebox.showerror('Error','Labels required'); return
        labels = read_yolo_labels(lbl_path)
        self.master = {'image': img_path, 'labels': labels}
        messagebox.showinfo('Master Loaded', f'Image: {os.path.basename(img_path)}\nLabels: {os.path.basename(lbl_path)}')

    def export_results(self):
        txt = self.feedback.get('1.0','end').strip()
        if not txt:
            messagebox.showinfo('Export','No results yet'); return
        path = filedialog.asksaveasfilename(defaultextension='.txt')
        if not path: return
        with open(path,'w',encoding='utf-8') as f: f.write(txt)
        messagebox.showinfo('Exported', f'Saved to {path}')

    def test_model(self):
        """Run model on the fixed test image Test2.jpg instead of the live frame."""
        if not hasattr(self, 'model') or not self.model:
            messagebox.showerror('Test Model', 'Model wrapper not initialized.'); return
        if self.model.model is None and not cfg.get('debug'):
            messagebox.showinfo('Test Model', 'No model loaded (and debug disabled). Train or place weights first.'); return
        # Resolve path (use given absolute path; fallback relative construction)
        test_path = r"C:\Users\User\OneDrive\Documents\Python Game\data\images\Test2.jpg"
        if not os.path.isfile(test_path):
            # fallback relative
            test_path_alt = os.path.join(cfg.get('data_dir','data'),'images','Test2.jpg')
            if os.path.isfile(test_path_alt):
                test_path = test_path_alt
        if not os.path.isfile(test_path):
            messagebox.showerror('Test Model', f'Test image not found: {test_path}')
            return
        frame = cv2.imread(test_path)
        if frame is None:
            messagebox.showerror('Test Model', 'Failed to read test image.'); return
        # Run prediction
        try:
            dets = self.model.predict(frame, img_size=cfg.get('img_size',640)) or []
        except Exception as e:
            messagebox.showerror('Test Model', f'Prediction failed: {e}'); return
        # Draw detections
        disp = frame.copy()
        for d in dets:
            try:
                x1,y1,x2,y2 = d['bbox']
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(disp,f"{d['class']}:{d['score']:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            except Exception:
                continue
        # Resize to fit preview window constraints
        target_w = cfg.get('preview_max_width', 960)
        target_h = cfg.get('preview_max_height', 720)
        h,w = disp.shape[:2]
        scale = min(target_w/float(w), target_h/float(h), 1.0)
        if scale < 1.0:
            disp_rs = cv2.resize(disp, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            disp_rs = disp
        rgb = cv2.cvtColor(disp_rs, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        win = tk.Toplevel(self.root)
        win.title('Test Model Result (Test2.jpg)')
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=6, pady=6)
        ttk.Label(win, text=f'Image: {os.path.basename(test_path)}  |  Detections: {len(dets)}').pack(pady=(0,4))
        ttk.Button(win, text='Close', command=win.destroy).pack(pady=4)

    def _on_detection_update(self, frame, detections, matches, feedback_lines):
        """UI update callback passed into DetectionEngine."""
        # Draw overlay on copy of original frame
        disp = frame.copy()
        for d in detections:
            try:
                x1,y1,x2,y2 = d['bbox']
                cv2.rectangle(disp,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(disp,f"{d['class']}:{d['score']:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            except Exception:
                continue
        for m in matches:
            try:
                verdict = getattr(m,'verdict', None) if not isinstance(m, dict) else m.get('verdict')
                bbox = getattr(m,'master_bbox', None) if not isinstance(m, dict) else m.get('master_bbox')
                if bbox:
                    color = (0,255,0) if verdict=='match' else (0,255,255) if verdict=='near' else (0,0,255)
                    x1,y1,x2,y2 = map(int, bbox)
                    cv2.rectangle(disp,(x1,y1),(x2,y2),color,2)
            except Exception:
                continue
        # Letterbox resize into fixed preview area
        target_w = getattr(self, '_main_prev_w', cfg.get('preview_max_width', 960))
        target_h = getattr(self, '_main_prev_h', cfg.get('preview_max_height', 720))
        h, w = disp.shape[:2]
        if w > 0 and h > 0:
            scale = min(target_w / float(w), target_h / float(h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            if (new_w, new_h) != (w, h):
                try:
                    disp_resized = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    disp_resized = disp
            else:
                disp_resized = disp
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            ox, oy = (target_w - new_w) // 2, (target_h - new_h) // 2
            canvas[oy:oy+new_h, ox:ox+new_w] = disp_resized
        else:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_panel.configure(image=tkimg); self.video_panel.image = tkimg
        # Feedback text
        self.feedback.configure(state='normal')
        self.feedback.delete('1.0','end')
        for line in feedback_lines:
            self.feedback.insert('end', line+'\n')
        self.feedback.configure(state='disabled')

    # ---------- Shutdown ----------
    def on_close(self):
        try: self.stop()
        except Exception: pass
        try: self._stop_webcam_preview()
        except Exception: pass
        for win_attr in ('_webcam_settings_window','_object_classification_window','_chatbot_settings_window'):
            win = getattr(self, win_attr, None)
            if win and win.winfo_exists():
                try: win.destroy()
                except Exception: pass
        # master trainer window
        if getattr(self,'_master_trainer_win', None) is not None:
            try:
                if self._master_trainer_win.winfo_exists():
                    self._master_trainer_win.destroy()
            except Exception: pass
            self._master_trainer_win=None
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()
