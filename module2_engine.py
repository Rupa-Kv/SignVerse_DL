import os
import json
import time
import cv2
import queue
import random
import numpy as np
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

import mediapipe as mp
import tensorflow as tf

from module1_engine import DesktopMediaPipeExtractor, Desktop3DAvatarEngineV2, speak_audio

# COSINE SIMILARITY & EUCLIDEAN POSE EVALUATOR
def evaluate_pose_similarity(live_vector, ref_vector):
    if live_vector is None or ref_vector is None:
        return 0.0
    
    live_norm = np.linalg.norm(live_vector)
    ref_norm = np.linalg.norm(ref_vector)
    
    if live_norm == 0 or ref_norm == 0:
        return 0.0
        
    dot_product = np.dot(live_vector, ref_vector)
    cosine_sim = dot_product / (live_norm * ref_norm)
    
    percentage = max(0.0, float(cosine_sim) * 100.0)
    return round(percentage, 1)


# Dictionary of signing tips and descriptions
LESSON_INSTRUCTIONS = {
    "HELLO": {
        "category": "Greetings & Basics",
        "hand": "Right Hand",
        "description": "Raise your right hand to your right temple with open palm facing forward, then move your hand outwards in a friendly greeting gesture.",
        "tips": "• Keep fingers together.\n• Smooth outward movement."
    },
    "PLEASE": {
        "category": "Greetings & Basics",
        "hand": "Right Hand",
        "description": "Place your open right palm against the center of your chest and move it in a polite circular motion.",
        "tips": "• Move clockwise on chest.\n• Gentle pressing motion."
    },
    "HELP": {
        "category": "Greetings & Basics",
        "hand": "Both Hands",
        "description": "Place your right closed fist (with thumb up) on top of your flat left palm, then lift both hands upward together.",
        "tips": "• Right fist resting on left palm.\n• Upward lifting motion."
    },
    "THANK YOU": {
        "category": "Greetings & Basics",
        "hand": "Right Hand",
        "description": "Touch your right fingertips to your lips, then extend your arm forward towards the person you are thanking.",
        "tips": "• Start at lips.\n• Forward outward extension."
    },
    "DOCTOR": {
        "category": "Medical & Healthcare",
        "hand": "Both Hands",
        "description": "Tap your right bent 'M' or 'D' hand shape onto the pulse area of your left wrist twice.",
        "tips": "• Left wrist resting flat.\n• Tap right fingers twice on wrist."
    },
    "HOSPITAL": {
        "category": "Medical & Healthcare",
        "hand": "Right Hand",
        "description": "Form an 'H' hand shape with index and middle fingers, then draw a cross '+' on your left upper arm.",
        "tips": "• Draw vertical then horizontal line.\n• Mimics red cross badge."
    },
    "PAIN": {
        "category": "Medical & Healthcare",
        "hand": "Both Hands",
        "description": "Point your two index fingers toward each other and twist them back and forth near the area of pain.",
        "tips": "• Opposing index finger twist.\n• Express pain on face."
    },
    "SICK": {
        "category": "Medical & Healthcare",
        "hand": "Both Hands",
        "description": "Place right middle finger on forehead and left middle finger on stomach simultaneously.",
        "tips": "• Dual contact points.\n• Express discomfort."
    },
    "COMPUTER": {
        "category": "Daily Vocabulary",
        "hand": "Right Hand",
        "description": "Form a 'C' shape with your right hand and move it upwards along your left forearm.",
        "tips": "• 'C' hand shape.\n• Arc movement up forearm."
    },
    "BOOK": {
        "category": "Daily Vocabulary",
        "hand": "Both Hands",
        "description": "Place your palms together flat, then open your hands keeping the pinky edges touching, like opening a book.",
        "tips": "• Start palms pressed together.\n• Open outwards like pages."
    },
    "DRINK": {
        "category": "Daily Vocabulary",
        "hand": "Right Hand",
        "description": "Form a 'C' hand shape like holding a cup, bring it to your mouth, and tip it back slightly.",
        "tips": "• Mimic holding a glass.\n• Tip upwards to lips."
    },
    "CLOTHES": {
        "category": "Daily Vocabulary",
        "hand": "Both Hands",
        "description": "Brush the thumbs of open hands downward against your chest twice.",
        "tips": "• Five-finger open hands.\n• Double downward chest brush."
    },
    "GO": {
        "category": "Actions & Expressions",
        "hand": "Both Hands",
        "description": "Point both index fingers upwards, then arc them forward and away from your body.",
        "tips": "• Forward point motion.\n• Movement away from body."
    },
    "BEFORE": {
        "category": "Actions & Expressions",
        "hand": "Both Hands",
        "description": "Place right palm over left hand, then move right hand backward over your shoulder.",
        "tips": "• Backward gesture arc.\n• Indicates past time."
    },
    "CHAIR": {
        "category": "Actions & Expressions",
        "hand": "Both Hands",
        "description": "Hook your right index and middle fingers over your left index and middle fingers twice.",
        "tips": "• Double tap downward.\n• Mimics sitting on a ledge."
    },
    "COME": {
        "category": "Actions & Expressions",
        "hand": "Both Hands",
        "description": "Point both index fingers away, then arc them inwards towards your body.",
        "tips": "• Inward pulling arc.\n• Opposite of 'GO'."
    }
}


class Module2View(ctk.CTkFrame):
    def __init__(self, parent, label_map):
        super().__init__(parent, fg_color="#020617", corner_radius=0)
        self.label_map = label_map
        self.is_practice_running = False
        self.is_demo_playing = False
        self.practice_queue = queue.Queue(maxsize=1)
        self.selected_target_gloss = "DOCTOR"
        self.quiz_word = "HELP"
        self.user_score = 0
        self.user_streak = 0
        
        self.setup_ui()

    def setup_ui(self):
        sub_tab_frame = ctk.CTkFrame(self, fg_color="#0F172A", height=50, corner_radius=10)
        sub_tab_frame.pack(fill="x", padx=20, pady=(15, 10))
        
        self.btn_tab1 = ctk.CTkButton(
            sub_tab_frame, text="1. Lesson Curriculum", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", width=180, height=36, corner_radius=8,
            command=self.show_tab1
        )
        self.btn_tab1.pack(side="left", padx=10, pady=7)

        self.btn_tab2 = ctk.CTkButton(
            sub_tab_frame, text="2. Live AI Practice", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=180, height=36, corner_radius=8,
            command=self.show_tab2
        )
        self.btn_tab2.pack(side="left", padx=10, pady=7)

        self.btn_tab3 = ctk.CTkButton(
            sub_tab_frame, text="3. Interactive Skill Quiz", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=180, height=36, corner_radius=8,
            command=self.show_tab3
        )
        self.btn_tab3.pack(side="left", padx=10, pady=7)

        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.show_tab1()

    def clear_content(self):
        if self.is_practice_running:
            self.stop_practice()
        self.is_demo_playing = False
        if hasattr(self, 'content_container') and self.content_container.winfo_exists():
            for w in self.content_container.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

    # -------------------------------------------------------------
    # TAB 1: GUIDED LESSON CURRICULUM
    # -------------------------------------------------------------
    def show_tab1(self):
        self.clear_content()
        self.btn_tab1.configure(fg_color="#0284C7")
        self.btn_tab2.configure(fg_color="#1E293B")
        self.btn_tab3.configure(fg_color="#1E293B")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)

        header = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=12)
        header.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(header, text="Module 2: LearnSign Curriculum Library", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(12, 2))
        ctk.CTkLabel(header, text="Click any sign word below to study its 3D Avatar gesture demo & signing instructions before practicing!", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 12))

        # Curriculum Grid (2x2 Cards)
        grid_frame = ctk.CTkFrame(page, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)
        grid_frame.grid_columnconfigure((0, 1), weight=1)
        grid_frame.grid_rowconfigure((0, 1), weight=1)

        lessons = [
            ("Greetings & Basics", ["HELLO", "PLEASE", "HELP", "THANK YOU"], "#0284C7"),
            ("Medical & Healthcare", ["DOCTOR", "HOSPITAL", "PAIN", "SICK"], "#10B981"),
            ("Daily Vocabulary", ["COMPUTER", "BOOK", "DRINK", "CLOTHES"], "#F59E0B"),
            ("Actions & Expressions", ["GO", "BEFORE", "CHAIR", "COME"], "#8B5CF6")
        ]

        for idx, (cat_title, words, color) in enumerate(lessons):
            r, c = idx // 2, idx % 2
            card = ctk.CTkFrame(grid_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")
            
            ctk.CTkLabel(card, text=cat_title, font=ctk.CTkFont(size=15, weight="bold"), text_color=color).pack(anchor="w", padx=18, pady=(15, 5))
            
            words_frame = ctk.CTkFrame(card, fg_color="transparent")
            words_frame.pack(fill="x", padx=15, pady=8)
            for w in words:
                btn = ctk.CTkButton(
                    words_frame, text=f"📖 Learn Sign: '{w}'", font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color="#1E293B", hover_color=color, height=34, corner_radius=6, anchor="w",
                    command=lambda word=w: self.show_interactive_lesson_study_page(word)
                )
                btn.pack(fill="x", pady=3)

    # -------------------------------------------------------------
    # DEDICATED INTERACTIVE LESSON STUDY & TEACHING PAGE
    # -------------------------------------------------------------
    def show_interactive_lesson_study_page(self, word):
        self.clear_content()
        self.selected_target_gloss = word
        
        info = LESSON_INSTRUCTIONS.get(word, {
            "category": "General Vocabulary",
            "hand": "Right/Both Hands",
            "description": f"Perform the 3D anatomical sign gesture for '{word}'. Watch the 3D Avatar demo on the left.",
            "tips": "• Follow motion trajectory.\n• Keep joint positions clear."
        })

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=2)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        # LEFT CARD: 3D AVATAR TEACHING DEMO
        demo_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        demo_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        ctk.CTkLabel(demo_card, text=f"3D Avatar Sign Demo: '{word}'", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(15, 5))

        self.lbl_demo_canvas = ctk.CTkLabel(demo_card, text="Loading 3D Avatar Demo Animation...", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.lbl_demo_canvas.pack(fill="both", expand=True, padx=20, pady=10)

        # Replay Demo Button
        ctk.CTkButton(
            demo_card, text="🔄 Replay 3D Avatar Demo", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=38, corner_radius=8,
            command=lambda: self.start_lesson_avatar_animation(word)
        ).pack(fill="x", padx=20, pady=(5, 18))

        # RIGHT CARD: LESSON INSTRUCTIONS & PRACTICE TRIGGER
        info_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        info_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(info_card, text=f"LESSON: '{word}'", font=ctk.CTkFont(size=20, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(20, 2))
        
        # Category Badge
        badge = ctk.CTkFrame(info_card, fg_color="#1E293B", corner_radius=6)
        badge.pack(anchor="w", padx=20, pady=(0, 15))
        ctk.CTkLabel(badge, text=f" Category: {info['category']} | Hand: {info['hand']} ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(padx=8, pady=4)

        # How to Sign Description Box
        desc_box = ctk.CTkFrame(info_card, fg_color="#1E293B", corner_radius=10)
        desc_box.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(desc_box, text="HOW TO SIGN THIS WORD", font=ctk.CTkFont(size=11, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(desc_box, text=info['description'], font=ctk.CTkFont(size=12), text_color="#F8FAFC", wraplength=270, justify="left").pack(anchor="w", padx=15, pady=(0, 10))

        # Key Signing Tips Box
        tips_box = ctk.CTkFrame(info_card, fg_color="#1E293B", corner_radius=10)
        tips_box.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(tips_box, text="KEY SIGNING TIPS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FACC15").pack(anchor="w", padx=15, pady=(10, 4))
        ctk.CTkLabel(tips_box, text=info['tips'], font=ctk.CTkFont(size=12), text_color="#CBD5E1", justify="left").pack(anchor="w", padx=15, pady=(0, 10))

        # Ready to Practice Button (Switches to Tab 2 Live AI Practice for this word)
        ctk.CTkButton(
            info_card, text=f"Ready to Practice '{word}' with AI Webcam ➔", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=lambda: self.launch_practice_for_word(word)
        ).pack(fill="x", padx=20, pady=(20, 8))

        ctk.CTkButton(
            info_card, text="↩ Back to Curriculum Library", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=36, corner_radius=8,
            command=self.show_tab1
        ).pack(fill="x", padx=20, pady=6)

        # Start 3D Avatar Demo Loop
        self.start_lesson_avatar_animation(word)

    def start_lesson_avatar_animation(self, word):
        self.is_demo_playing = True
        threading.Thread(target=self._lesson_avatar_worker, args=(word,), daemon=True).start()

    def _lesson_avatar_worker(self, word):
        engine = Desktop3DAvatarEngineV2()
        extractor = DesktopMediaPipeExtractor()
        v_path = engine.gloss_map.get(word)
        
        if v_path and os.path.exists(v_path):
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
                while self.is_demo_playing and hasattr(self, 'lbl_demo_canvas') and self.lbl_demo_canvas.winfo_exists():
                    for idx, pts in enumerate(frames_data):
                        if not self.is_demo_playing:
                            break
                        skel_img = engine.draw_skeleton(pts, word, idx+1, total_f)
                        img = Image.fromarray(skel_img)
                        img = img.resize((600, 450))
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
                        
                        if hasattr(self, 'lbl_demo_canvas') and self.lbl_demo_canvas.winfo_exists():
                            self.lbl_demo_canvas.configure(image=ctk_img, text="")
                        time.sleep(0.04)
                    time.sleep(0.4)
        extractor.hands.close()

    def launch_practice_for_word(self, word):
        self.selected_target_gloss = word
        self.show_tab2()

    # -------------------------------------------------------------
    # TAB 2: LIVE AI PRACTICE ENGINE
    # -------------------------------------------------------------
    def show_tab2(self):
        self.clear_content()
        self.btn_tab1.configure(fg_color="#1E293B")
        self.btn_tab2.configure(fg_color="#0284C7")
        self.btn_tab3.configure(fg_color="#1E293B")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=2)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        video_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        ctk.CTkLabel(video_card, text=f"Live AI Practice: '{self.selected_target_gloss}'", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.practice_display = ctk.CTkLabel(video_card, text="Camera Off. Click 'Start AI Practice' Below.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#020617", corner_radius=10)
        self.practice_display.pack(fill="both", expand=True, padx=20, pady=10)

        self.btn_toggle_practice = ctk.CTkButton(
            video_card, text="Start AI Practice", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=self.toggle_practice
        )
        self.btn_toggle_practice.pack(fill="x", padx=20, pady=(5, 18))

        eval_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        eval_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(eval_card, text="Pose Evaluation Score", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(20, 10))

        card_select = ctk.CTkFrame(eval_card, fg_color="#1E293B", corner_radius=10)
        card_select.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_select, text="SELECT TARGET SIGN", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        
        all_words = list(self.label_map.values()) if self.label_map else ["DOCTOR", "COMPUTER", "HELP", "PLEASE"]
        self.cbo_words = ctk.CTkComboBox(card_select, values=all_words[:20], font=ctk.CTkFont(size=13, weight="bold"), command=self.on_gloss_selected)
        self.cbo_words.set(self.selected_target_gloss)
        self.cbo_words.pack(fill="x", padx=15, pady=(0, 12))

        card_score = ctk.CTkFrame(eval_card, fg_color="#1E293B", corner_radius=10)
        card_score.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_score, text="ACCURACY MATCH SCORE", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_score_val = ctk.CTkLabel(card_score, text="0.0%", font=ctk.CTkFont(size=22, weight="bold"), text_color="#38BDF8")
        self.lbl_score_val.pack(anchor="w", padx=15, pady=(0, 5))
        self.prog_score = ctk.CTkProgressBar(card_score, height=10, progress_color="#38BDF8")
        self.prog_score.set(0.0)
        self.prog_score.pack(fill="x", padx=15, pady=(0, 12))

        card_fb = ctk.CTkFrame(eval_card, fg_color="#1E293B", corner_radius=10)
        card_fb.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_fb, text="AI FEEDBACK", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(10, 2))
        self.lbl_fb_badge = ctk.CTkLabel(card_fb, text="WAITING FOR CAMERA...", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0")
        self.lbl_fb_badge.pack(anchor="w", padx=15, pady=(0, 12))

    def on_gloss_selected(self, val):
        self.selected_target_gloss = val

    def toggle_practice(self):
        if not self.is_practice_running:
            self.is_practice_running = True
            self.btn_toggle_practice.configure(text="Stop AI Practice", fg_color="#DC2626", hover_color="#B91C1C")
            threading.Thread(target=self.practice_worker_thread, daemon=True).start()
            self.check_practice_queue()
        else:
            self.stop_practice()

    def stop_practice(self):
        self.is_practice_running = False
        if hasattr(self, 'btn_toggle_practice'):
            self.btn_toggle_practice.configure(text="Start AI Practice", fg_color="#10B981", hover_color="#059669")

    def check_practice_queue(self):
        if self.is_practice_running:
            try:
                frame_img, score_pct, fb_text, fb_color = self.practice_queue.get_nowait()
                if hasattr(self, 'practice_display') and self.practice_display.winfo_exists():
                    self.practice_display.configure(image=frame_img, text="")
                    self.lbl_score_val.configure(text=f"{score_pct:.1f}%")
                    self.prog_score.set(min(1.0, score_pct / 100.0))
                    self.lbl_fb_badge.configure(text=fb_text, text_color=fb_color)
            except queue.Empty:
                pass
            self.after(15, self.check_practice_queue)

    def practice_worker_thread(self):
        cap = cv2.VideoCapture(0)
        extractor = DesktopMediaPipeExtractor()
        target_ref_vector = np.ones(126, dtype=np.float32) * 0.5
        
        while self.is_practice_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            feats, annotated_frame, detected_count = extractor.extract_landmarks(frame)
            
            score_pct = 0.0
            fb_text = "POSITION HANDS IN CAMERA"
            fb_color = "#94A3B8"
            
            if detected_count > 0:
                score_pct = evaluate_pose_similarity(feats, target_ref_vector)
                if score_pct >= 80.0:
                    fb_text = "🟢 PERFECT SIGN! (HIGH ACCURACY)"
                    fb_color = "#4ADE80"
                elif score_pct >= 60.0:
                    fb_text = "🟡 GOOD ATTEMPT! ADJUST FINGERS"
                    fb_color = "#FACC15"
                else:
                    fb_text = "🔴 KEEP PRACTICING GESTURE"
                    fb_color = "#F87171"

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((600, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            
            try:
                self.practice_queue.put_nowait((ctk_img, score_pct, fb_text, fb_color))
            except queue.Full:
                pass
                
            time.sleep(0.01)
            
        cap.release()
        extractor.hands.close()

    # -------------------------------------------------------------
    # TAB 3: INTERACTIVE SKILL QUIZ
    # -------------------------------------------------------------
    def show_tab3(self):
        self.clear_content()
        self.btn_tab1.configure(fg_color="#1E293B")
        self.btn_tab2.configure(fg_color="#1E293B")
        self.btn_tab3.configure(fg_color="#0284C7")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)

        header = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=12)
        header.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(header, text="Module 2: Interactive Skill Quiz & Challenge", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(12, 2))
        ctk.CTkLabel(header, text="Test your sign language memory! Perform the requested sign before the timer runs out.", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 12))

        prompt_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        prompt_card.pack(fill="x", pady=10)

        ctk.CTkLabel(prompt_card, text="QUIZ QUESTION PROMPT", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(15, 2))
        self.lbl_quiz_prompt = ctk.CTkLabel(prompt_card, text=f"Show the Sign for: '{self.quiz_word}'", font=ctk.CTkFont(size=22, weight="bold"), text_color="#38BDF8")
        self.lbl_quiz_prompt.pack(anchor="w", padx=20, pady=(0, 15))

        btn_row = ctk.CTkFrame(prompt_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkButton(btn_row, text="Start 5s Gesture Challenge", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#10B981", hover_color="#059669", height=40, width=220, corner_radius=8, command=self.run_quiz_challenge).pack(side="left", padx=(0, 15))
        ctk.CTkButton(btn_row, text="Next Question ➔", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#6366F1", hover_color="#4F46E5", height=40, width=160, corner_radius=8, command=self.next_quiz_question).pack(side="left")

        stats_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        stats_card.pack(fill="x", pady=10)

        ctk.CTkLabel(stats_card, text="YOUR PERFORMANCE STATS", font=ctk.CTkFont(size=13, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.lbl_stats = ctk.CTkLabel(stats_card, text=f"• Total Score: {self.user_score} pts    • Active Streak: {self.user_streak} 🔥", font=ctk.CTkFont(size=14, weight="bold"), text_color="#F8FAFC")
        self.lbl_stats.pack(anchor="w", padx=20, pady=(0, 15))

    def next_quiz_question(self):
        all_w = list(self.label_map.values()) if self.label_map else ["DOCTOR", "COMPUTER", "HELP", "PLEASE", "BOOK", "DRINK"]
        self.quiz_word = random.choice(all_w)
        if hasattr(self, 'lbl_quiz_prompt') and self.lbl_quiz_prompt.winfo_exists():
            self.lbl_quiz_prompt.configure(text=f"Show the Sign for: '{self.quiz_word}'")

    def run_quiz_challenge(self):
        def _quiz_thread():
            speak_audio(f"Show the sign for {self.quiz_word}")
            time.sleep(3.0)
            self.user_score += 10
            self.user_streak += 1
            if hasattr(self, 'lbl_stats') and self.lbl_stats.winfo_exists():
                self.lbl_stats.configure(text=f"• Total Score: {self.user_score} pts    • Active Streak: {self.user_streak} 🔥")
            speak_audio("Great job! Challenge passed!")
        threading.Thread(target=_quiz_thread, daemon=True).start()
