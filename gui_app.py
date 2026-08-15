import os
import json
import time
import cv2
import queue
import numpy as np
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

import mediapipe as mp
import tensorflow as tf

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# System Theme Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# MediaPipe 3D Landmark Extractor Class
class DesktopMediaPipeExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _normalize_hand(self, pts):
        wrist = pts[0, :].copy()
        pts = pts - wrist
        max_val = np.max(np.abs(pts))
        if max_val > 0:
            pts = pts / max_val
        return pts

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks_vector = np.zeros(2 * 21 * 3, dtype=np.float32)
        annotated_image = image.copy()
        detected_count = 0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            detected_count = len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                pts = self._normalize_hand(pts)
                pts_flat = pts.flatten()
                
                if label == 'Left':
                    landmarks_vector[0:63] = pts_flat
                else:
                    landmarks_vector[63:126] = pts_flat
                    
        return landmarks_vector, annotated_image, detected_count

# 3D Avatar Renderer Engine Class
class Desktop3DAvatarEngine:
    def __init__(self, dataset_dir="C:/Users/Rupa/Documents/SignVerse/DATASET_WLASL"):
        self.wlasl_json = os.path.join(dataset_dir, "WLASL_v0.3.json")
        self.video_dir = os.path.join(dataset_dir, "videos")
        self.gloss_map = {}
        
        if os.path.exists(self.wlasl_json):
            with open(self.wlasl_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    gloss = entry['gloss'].upper()
                    for inst in entry['instances']:
                        v_id = inst['video_id']
                        v_path = os.path.join(self.video_dir, f"{v_id}.mp4")
                        if os.path.exists(v_path):
                            self.gloss_map[gloss] = v_path
                            break

    def draw_skeleton(self, pts_vector, gloss_name, frame_num, total_frames):
        canvas = np.zeros((450, 600, 3), dtype=np.uint8)
        canvas.fill(20)
        h, w = canvas.shape[:2]
        
        cv2.rectangle(canvas, (0, 0), (w, 55), (35, 35, 35), -1)
        cv2.putText(canvas, f"SignVerse 3D Avatar: {gloss_name.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(canvas, f"Frame {frame_num}/{total_frames} | Skeleton Joint Motion", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        head_center = (w // 2, 105)
        neck = (w // 2, 140)
        left_shoulder = (w // 2 - 70, 165)
        right_shoulder = (w // 2 + 70, 165)
        spine_base = (w // 2, 245)
        
        cv2.circle(canvas, head_center, 28, (0, 200, 255), 3)
        cv2.line(canvas, head_center, neck, (200, 200, 200), 3)
        cv2.line(canvas, left_shoulder, right_shoulder, (200, 200, 200), 4)
        cv2.line(canvas, neck, spine_base, (200, 200, 200), 3)
        
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        ]
        
        left_pts = pts_vector[0:63].reshape((21, 3))
        if np.any(left_pts):
            l_wrist_px = (int(left_shoulder[0] + left_pts[0,0]*110), int(left_shoulder[1] + 55 + left_pts[0,1]*110))
            cv2.line(canvas, left_shoulder, l_wrist_px, (100, 255, 100), 3)
            l_px_list = []
            for pt in left_pts:
                px = (int(l_wrist_px[0] + pt[0]*90), int(l_wrist_px[1] + pt[1]*90))
                l_px_list.append(px)
                cv2.circle(canvas, px, 3, (0, 255, 0), -1)
            for p1, p2 in connections:
                cv2.line(canvas, l_px_list[p1], l_px_list[p2], (50, 200, 50), 2)
                
        right_pts = pts_vector[63:126].reshape((21, 3))
        if np.any(right_pts):
            r_wrist_px = (int(right_shoulder[0] + right_pts[0,0]*110), int(right_shoulder[1] + 55 + right_pts[0,1]*110))
            cv2.line(canvas, right_shoulder, r_wrist_px, (255, 150, 100), 3)
            r_px_list = []
            for pt in right_pts:
                px = (int(r_wrist_px[0] + pt[0]*90), int(r_wrist_px[1] + pt[1]*90))
                r_px_list.append(px)
                cv2.circle(canvas, px, 3, (255, 100, 0), -1)
            for p1, p2 in connections:
                cv2.line(canvas, r_px_list[p1], r_px_list[p2], (220, 100, 0), 2)
                
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Audio Speech Engine Function
def speak_audio(text):
    if TTS_AVAILABLE and text:
        def _speak():
            try:
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_speak, daemon=True).start()

# Modern CustomTkinter Desktop Application Class
class ModernSignVerseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SignVerse AI Desktop Platform")
        self.geometry("1150x720")
        self.minsize(1000, 650)
        
        # Load Trained Model
        self.model = None
        self.label_map = {}
        self.load_model()
        
        # Camera State & Queue (Thread-Safe Flicker Prevention)
        self.is_camera_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_words = []
        self.current_page = "sign_to_text"
        
        self.setup_ui_layout()
        self.show_sign_to_text_page()

    def load_model(self):
        model_path = "saved_models/best_signverse_model_300.keras"
        label_path = "preprocessed_data_300/label_map_300.json"
        if os.path.exists(model_path) and os.path.exists(label_path):
            self.model = tf.keras.models.load_model(model_path)
            with open(label_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
                self.label_map = {int(k): v for k, v in self.label_map.items()}

    def setup_ui_layout(self):
        # Configure Grid Layout (2 Columns: Sidebar + Main Content)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # LEFT SIDEBAR NAVIGATION
        # -------------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#0F172A")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SignVerse AI", font=ctk.CTkFont(size=22, weight="bold"), text_color="#00E5FF")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 5), sticky="w")
        
        self.sub_logo = ctk.CTkLabel(self.sidebar_frame, text="Communication Engine", font=ctk.CTkFont(size=12), text_color="#94A3B8")
        self.sub_logo.grid(row=1, column=0, padx=20, pady=(0, 25), sticky="w")

        # Sidebar Buttons
        self.btn_nav_opt1 = ctk.CTkButton(
            self.sidebar_frame, 
            text="Option 1: Sign-to-Text / Audio", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", 
            hover_color="#0369A1",
            height=40,
            corner_radius=8,
            anchor="w",
            command=self.show_sign_to_text_page
        )
        self.btn_nav_opt1.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.btn_nav_opt2 = ctk.CTkButton(
            self.sidebar_frame, 
            text="Option 2: Text-to-3D Avatar", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", 
            hover_color="#334155",
            height=40,
            corner_radius=8,
            anchor="w",
            command=self.show_text_to_sign_page
        )
        self.btn_nav_opt2.grid(row=3, column=0, padx=15, pady=8, sticky="ew")

        # System Info Box at bottom of sidebar
        self.info_box = ctk.CTkFrame(self.sidebar_frame, fg_color="#1E293B", corner_radius=10)
        self.info_box.grid(row=6, column=0, padx=15, pady=20, sticky="ew")
        
        ctk.CTkLabel(self.info_box, text="System Specifications", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(self.info_box, text="• Benchmark: WLASL-300\n• Features: 126 Landmarks\n• Model: Conv1D + BiLSTM\n• Speech: Windows SAPI5", font=ctk.CTkFont(size=11), text_color="#94A3B8", justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        # -------------------------------------------------------------
        # RIGHT MAIN CONTENT CONTAINER
        # -------------------------------------------------------------
        self.main_container = ctk.CTkFrame(self, fg_color="#020617", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

    def clear_main_container(self):
        if self.is_camera_running:
            self.stop_camera()
        for widget in self.main_container.winfo_children():
            widget.destroy()

    # -------------------------------------------------------------
    # PAGE 1: SIGN TO TEXT / AUDIO (LIVE CAMERA)
    # -------------------------------------------------------------
    def show_sign_to_text_page(self):
        self.clear_main_container()
        self.current_page = "sign_to_text"
        self.btn_nav_opt1.configure(fg_color="#0284C7")
        self.btn_nav_opt2.configure(fg_color="#1E293B")

        page_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        page_frame.pack(fill="both", expand=True, padx=25, pady=25)
        page_frame.grid_columnconfigure(0, weight=2)
        page_frame.grid_columnconfigure(1, weight=1)
        page_frame.grid_rowconfigure(0, weight=1)

        # LEFT CARD: Video Stream Window
        video_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        ctk.CTkLabel(video_card, text="Option 1: Live Sign Language Camera Recognizer", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.video_display = ctk.CTkLabel(video_card, text="Camera Off. Click 'Start Live Stream' Below.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.video_display.pack(fill="both", expand=True, padx=20, pady=10)

        self.btn_toggle_cam = ctk.CTkButton(
            video_card, 
            text="Start Live Stream", 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", 
            hover_color="#059669",
            height=45,
            corner_radius=10,
            command=self.toggle_camera
        )
        self.btn_toggle_cam.pack(fill="x", padx=20, pady=(5, 18))

        # RIGHT CARD: Live Predictions & Sentence Controls
        metrics_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        metrics_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(metrics_card, text="Live Prediction Metrics", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(20, 10))

        # Prediction Word Badge
        self.card_sign = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        self.card_sign.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(self.card_sign, text="PREDICTED SIGN", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_sign_val = ctk.CTkLabel(self.card_sign, text="WAITING...", font=ctk.CTkFont(size=20, weight="bold"), text_color="#38BDF8")
        self.lbl_sign_val.pack(anchor="w", padx=15, pady=(0, 10))

        # Confidence Bar
        self.card_conf = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        self.card_conf.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(self.card_conf, text="CONFIDENCE SCORE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_conf_val = ctk.CTkLabel(self.card_conf, text="0.0%", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4ADE80")
        self.lbl_conf_val.pack(anchor="w", padx=15, pady=(0, 5))
        self.progress_conf = ctk.CTkProgressBar(self.card_conf, height=8, progress_color="#4ADE80")
        self.progress_conf.set(0.0)
        self.progress_conf.pack(fill="x", padx=15, pady=(0, 12))

        # Sentence Builder Box
        self.card_sent = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        self.card_sent.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(self.card_sent, text="ACTIVE SENTENCE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_sent_val = ctk.CTkLabel(self.card_sent, text="[ Empty ]", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC", wraplength=260, justify="left")
        self.lbl_sent_val.pack(anchor="w", padx=15, pady=(0, 12))

        # Action Buttons
        ctk.CTkButton(
            metrics_card, 
            text="Speak Active Sentence", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#6366F1", 
            hover_color="#4F46E5",
            height=38,
            corner_radius=8,
            command=lambda: speak_audio(" ".join(self.current_words))
        ).pack(fill="x", padx=20, pady=(15, 6))

        ctk.CTkButton(
            metrics_card, 
            text="Clear Sentence Buffer", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#DC2626", 
            hover_color="#B91C1C",
            height=38,
            corner_radius=8,
            command=self.clear_sentence_buffer
        ).pack(fill="x", padx=20, pady=6)

    def clear_sentence_buffer(self):
        self.current_words = []
        if hasattr(self, 'lbl_sent_val'):
            self.lbl_sent_val.configure(text="[ Empty ]")

    def toggle_camera(self):
        if not self.is_camera_running:
            self.is_camera_running = True
            self.btn_toggle_cam.configure(text="Stop Live Stream", fg_color="#DC2626", hover_color="#B91C1C")
            threading.Thread(target=self.camera_worker_thread, daemon=True).start()
            self.check_camera_queue()
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_camera_running = False
        if hasattr(self, 'btn_toggle_cam'):
            self.btn_toggle_cam.configure(text="Start Live Stream", fg_color="#10B981", hover_color="#059669")

    # Thread-Safe Queue Check (Prevents Camera Blinking Completely)
    def check_camera_queue(self):
        if self.is_camera_running:
            try:
                frame_img, pred_sign, pred_conf, sent_text = self.frame_queue.get_nowait()
                self.video_display.configure(image=frame_img, text="")
                self.lbl_sign_val.configure(text=pred_sign.upper() if pred_sign else "WAITING...")
                self.lbl_conf_val.configure(text=f"{pred_conf*100:.1f}%")
                self.progress_conf.set(min(1.0, pred_conf))
                self.lbl_sent_val.configure(text=sent_text if sent_text else "[ Empty ]")
            except queue.Empty:
                pass
            self.after(15, self.check_camera_queue)

    def camera_worker_thread(self):
        cap = cv2.VideoCapture(0)
        extractor = DesktopMediaPipeExtractor()
        sequence_buffer = []
        last_prediction = ""
        prediction_confidence = 0.0
        last_hand_time = time.time()
        
        while self.is_camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            feats, annotated_frame, detected_count = extractor.extract_landmarks(frame)
            
            if detected_count > 0:
                last_hand_time = time.time()
            else:
                if self.current_words and (time.time() - last_hand_time > 2.0):
                    sentence_str = " ".join(self.current_words) + "."
                    speak_audio(sentence_str)
                    self.current_words = []
                    last_prediction = ""

            sequence_buffer.append(feats)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == 30 and detected_count > 0 and self.model:
                seq_input = np.array([sequence_buffer], dtype=np.float32)
                probs = self.model.predict(seq_input, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                prediction_confidence = float(probs[pred_idx])
                pred_word = self.label_map.get(pred_idx, "Unknown")
                
                if prediction_confidence > 0.65 and pred_word != last_prediction:
                    last_prediction = pred_word
                    if not self.current_words or self.current_words[-1] != pred_word:
                        self.current_words.append(pred_word)

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((600, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            
            sent_str = " ".join(self.current_words)
            try:
                self.frame_queue.put_nowait((ctk_img, last_prediction, prediction_confidence, sent_str))
            except queue.Full:
                pass
                
            time.sleep(0.01)
            
        cap.release()
        extractor.hands.close()

    # -------------------------------------------------------------
    # PAGE 2: TEXT TO SIGN (3D AVATAR)
    # -------------------------------------------------------------
    def show_text_to_sign_page(self):
        self.clear_main_container()
        self.current_page = "text_to_sign"
        self.btn_nav_opt1.configure(fg_color="#1E293B")
        self.btn_nav_opt2.configure(fg_color="#0284C7")

        page_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        page_frame.pack(fill="both", expand=True, padx=25, pady=25)

        # Header Box
        header_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        header_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(header_card, text="Option 2: Text-to-Sign 3D Skeleton Avatar Translator", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        input_row = ctk.CTkFrame(header_card, fg_color="transparent")
        input_row.pack(fill="x", padx=20, pady=(5, 15))

        self.txt_input = ctk.CTkEntry(input_row, placeholder_text="Enter text sentence...", font=ctk.CTkFont(size=13), height=40, corner_radius=8)
        self.txt_input.insert(0, "hello doctor please help computer")
        self.txt_input.pack(side="left", fill="x", expand=True, padx=(0, 15))

        ctk.CTkButton(
            input_row, 
            text="Render 3D Avatar", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981", 
            hover_color="#059669",
            height=40,
            width=160,
            corner_radius=8,
            command=self.start_avatar_rendering
        ).pack(side="right")

        # Avatar Display Canvas
        display_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        display_card.pack(fill="both", expand=True)

        self.avatar_display = ctk.CTkLabel(display_card, text="Enter text above and click 'Render 3D Avatar'.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.avatar_display.pack(fill="both", expand=True, padx=20, pady=20)

    def start_avatar_rendering(self):
        text = self.txt_input.get().strip()
        if text:
            threading.Thread(target=self.avatar_rendering_worker, args=(text,), daemon=True).start()

    def avatar_rendering_worker(self, text):
        engine = Desktop3DAvatarEngine()
        extractor = DesktopMediaPipeExtractor()
        words = [w.strip().upper() for w in text.split() if w.strip()]
        
        for word in words:
            v_path = engine.gloss_map.get(word)
            if not v_path:
                continue
                
            cap = cv2.VideoCapture(v_path)
            frames_data = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                vector, _, _ = extractor.extract_landmarks(frame)
                frames_data.append(vector)
            cap.release()
            
            if frames_data:
                total_f = len(frames_data)
                for idx, pts in enumerate(frames_data):
                    skel_img = engine.draw_skeleton(pts, word, idx+1, total_f)
                    img = Image.fromarray(skel_img)
                    img = img.resize((600, 450))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
                    
                    self.avatar_display.configure(image=ctk_img, text="")
                    time.sleep(0.03)
                time.sleep(0.3)
                
        extractor.hands.close()

if __name__ == "__main__":
    app = ModernSignVerseApp()
    app.mainloop()
