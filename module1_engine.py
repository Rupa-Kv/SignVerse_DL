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


# 3D ANATOMICAL AVATAR ENGINE CLASS
class Desktop3DAvatarEngineV2:
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
        canvas.fill(15)
        h, w = canvas.shape[:2]
        
        cv2.rectangle(canvas, (0, 0), (w, 55), (30, 30, 45), -1)
        cv2.putText(canvas, f"SignVerse 3D Avatar: {gloss_name.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 225, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame {frame_num}/{total_frames} | Anatomical Hand Mesh Simulation", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)
        
        head_center = (w // 2, 105)
        neck = (w // 2, 145)
        left_shoulder = (w // 2 - 75, 170)
        right_shoulder = (w // 2 + 75, 170)
        torso_base = (w // 2, 260)
        
        torso_poly = np.array([neck, left_shoulder, (left_shoulder[0] + 15, torso_base[1]), (right_shoulder[0] - 15, torso_base[1]), right_shoulder], np.int32)
        cv2.fillPoly(canvas, [torso_poly], (35, 45, 60), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [torso_poly], True, (70, 90, 120), 2, lineType=cv2.LINE_AA)

        cv2.circle(canvas, head_center, 30, (45, 55, 75), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, head_center, 30, (0, 200, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (w // 2, 135), neck, (70, 90, 120), 4, lineType=cv2.LINE_AA)
        
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        ]
        
        # Left Hand
        left_pts = pts_vector[0:63].reshape((21, 3))
        if np.any(left_pts):
            l_wrist_px = (int(left_shoulder[0] + left_pts[0,0]*110), int(left_shoulder[1] + 55 + left_pts[0,1]*110))
            cv2.line(canvas, left_shoulder, l_wrist_px, (0, 200, 150), 5, lineType=cv2.LINE_AA)
            l_px_list = []
            for pt in left_pts:
                px = (int(l_wrist_px[0] + pt[0]*90), int(l_wrist_px[1] + pt[1]*90))
                l_px_list.append(px)
            palm_pts_l = np.array([l_px_list[0], l_px_list[5], l_px_list[9], l_px_list[13], l_px_list[17]], np.int32)
            cv2.fillPoly(canvas, [palm_pts_l], (15, 65, 45), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [palm_pts_l], True, (0, 220, 140), 2, lineType=cv2.LINE_AA)
            for p1, p2 in connections:
                cv2.line(canvas, l_px_list[p1], l_px_list[p2], (0, 240, 160), 4, lineType=cv2.LINE_AA)
            for px in l_px_list:
                cv2.circle(canvas, px, 5, (0, 255, 180), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, px, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                
        # Right Hand
        right_pts = pts_vector[63:126].reshape((21, 3))
        if np.any(right_pts):
            r_wrist_px = (int(right_shoulder[0] + right_pts[0,0]*110), int(right_shoulder[1] + 55 + right_pts[0,1]*110))
            cv2.line(canvas, right_shoulder, r_wrist_px, (255, 120, 40), 5, lineType=cv2.LINE_AA)
            r_px_list = []
            for pt in right_pts:
                px = (int(r_wrist_px[0] + pt[0]*90), int(r_wrist_px[1] + pt[1]*90))
                r_px_list.append(px)
            palm_pts_r = np.array([r_px_list[0], r_px_list[5], r_px_list[9], r_px_list[13], r_px_list[17]], np.int32)
            cv2.fillPoly(canvas, [palm_pts_r], (70, 40, 15), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [palm_pts_r], True, (255, 140, 40), 2, lineType=cv2.LINE_AA)
            for p1, p2 in connections:
                cv2.line(canvas, r_px_list[p1], r_px_list[p2], (255, 160, 50), 4, lineType=cv2.LINE_AA)
            for px in r_px_list:
                cv2.circle(canvas, px, 5, (255, 180, 80), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, px, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


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


class Module1View(ctk.CTkFrame):
    def __init__(self, parent, model, label_map):
        super().__init__(parent, fg_color="#020617", corner_radius=0)
        self.model = model
        self.label_map = label_map
        self.is_camera_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_words = []
        
        self.setup_ui()

    def setup_ui(self):
        sub_tab_frame = ctk.CTkFrame(self, fg_color="#0F172A", height=50, corner_radius=10)
        sub_tab_frame.pack(fill="x", padx=20, pady=(15, 10))
        
        self.btn_opt1 = ctk.CTkButton(
            sub_tab_frame, text="Option 1: Sign-to-Text / Audio", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", width=220, height=36, corner_radius=8,
            command=self.show_opt1
        )
        self.btn_opt1.pack(side="left", padx=10, pady=7)

        self.btn_opt2 = ctk.CTkButton(
            sub_tab_frame, text="Option 2: Text-to-3D Avatar Mesh", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=220, height=36, corner_radius=8,
            command=self.show_opt2
        )
        self.btn_opt2.pack(side="left", padx=10, pady=7)

        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.show_opt1()

    def clear_content(self):
        if self.is_camera_running:
            self.stop_camera()
        if hasattr(self, 'content_container') and self.content_container.winfo_exists():
            for w in self.content_container.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

    def show_opt1(self):
        self.clear_content()
        self.btn_opt1.configure(fg_color="#0284C7")
        self.btn_opt2.configure(fg_color="#1E293B")

        page_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page_frame.pack(fill="both", expand=True)
        page_frame.grid_columnconfigure(0, weight=2)
        page_frame.grid_columnconfigure(1, weight=1)
        page_frame.grid_rowconfigure(0, weight=1)

        video_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        ctk.CTkLabel(video_card, text="Option 1: Live Sign Language Camera Recognizer", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.video_display = ctk.CTkLabel(video_card, text="Camera Off. Click 'Start Live Stream' Below.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.video_display.pack(fill="both", expand=True, padx=20, pady=10)

        self.btn_toggle_cam = ctk.CTkButton(
            video_card, text="Start Live Stream", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=self.toggle_camera
        )
        self.btn_toggle_cam.pack(fill="x", padx=20, pady=(5, 18))

        metrics_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        metrics_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(metrics_card, text="Live Prediction Metrics", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(20, 10))

        card_sign = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        card_sign.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_sign, text="PREDICTED SIGN", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_sign_val = ctk.CTkLabel(card_sign, text="WAITING...", font=ctk.CTkFont(size=20, weight="bold"), text_color="#38BDF8")
        self.lbl_sign_val.pack(anchor="w", padx=15, pady=(0, 10))

        card_conf = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        card_conf.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_conf, text="CONFIDENCE SCORE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_conf_val = ctk.CTkLabel(card_conf, text="0.0%", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4ADE80")
        self.lbl_conf_val.pack(anchor="w", padx=15, pady=(0, 5))
        self.progress_conf = ctk.CTkProgressBar(card_conf, height=8, progress_color="#4ADE80")
        self.progress_conf.set(0.0)
        self.progress_conf.pack(fill="x", padx=15, pady=(0, 12))

        card_sent = ctk.CTkFrame(metrics_card, fg_color="#1E293B", corner_radius=10)
        card_sent.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_sent, text="ACTIVE SENTENCE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_sent_val = ctk.CTkLabel(card_sent, text="[ Empty ]", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC", wraplength=260, justify="left")
        self.lbl_sent_val.pack(anchor="w", padx=15, pady=(0, 12))

        ctk.CTkButton(metrics_card, text="Speak Active Sentence", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#6366F1", hover_color="#4F46E5", height=38, corner_radius=8, command=lambda: speak_audio(" ".join(self.current_words))).pack(fill="x", padx=20, pady=(15, 6))
        ctk.CTkButton(metrics_card, text="Clear Sentence Buffer", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#DC2626", hover_color="#B91C1C", height=38, corner_radius=8, command=self.clear_sentence_buffer).pack(fill="x", padx=20, pady=6)

    def show_opt2(self):
        self.clear_content()
        self.btn_opt1.configure(fg_color="#1E293B")
        self.btn_opt2.configure(fg_color="#0284C7")

        page_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page_frame.pack(fill="both", expand=True)

        header_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        header_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(header_card, text="Option 2: Text-to-Sign 3D Anatomical Avatar Mesh Translator", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        input_row = ctk.CTkFrame(header_card, fg_color="transparent")
        input_row.pack(fill="x", padx=20, pady=(5, 15))

        self.txt_input = ctk.CTkEntry(input_row, placeholder_text="Enter text sentence...", font=ctk.CTkFont(size=13), height=40, corner_radius=8)
        self.txt_input.insert(0, "hello doctor please help computer")
        self.txt_input.pack(side="left", fill="x", expand=True, padx=(0, 15))

        ctk.CTkButton(input_row, text="Render 3D Avatar Mesh", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#10B981", hover_color="#059669", height=40, width=180, corner_radius=8, command=self.start_avatar_rendering).pack(side="right")

        display_card = ctk.CTkFrame(page_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        display_card.pack(fill="both", expand=True)

        self.avatar_display = ctk.CTkLabel(display_card, text="Enter text above and click 'Render 3D Avatar Mesh'.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.avatar_display.pack(fill="both", expand=True, padx=20, pady=20)

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

    def check_camera_queue(self):
        if self.is_camera_running:
            try:
                frame_img, pred_sign, pred_conf, sent_text = self.frame_queue.get_nowait()
                if hasattr(self, 'video_display') and self.video_display.winfo_exists():
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
                
                target_threshold = 0.55 if detected_count == 2 else 0.65
                
                if prediction_confidence > target_threshold and pred_word != last_prediction:
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

    def start_avatar_rendering(self):
        text = self.txt_input.get().strip()
        if text:
            threading.Thread(target=self.avatar_rendering_worker, args=(text,), daemon=True).start()

    def avatar_rendering_worker(self, text):
        engine = Desktop3DAvatarEngineV2()
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
                    
                    if hasattr(self, 'avatar_display') and self.avatar_display.winfo_exists():
                        self.avatar_display.configure(image=ctk_img, text="")
                    time.sleep(0.03)
                time.sleep(0.3)
                
        extractor.hands.close()
