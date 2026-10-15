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


# PERSISTENT USER PROFILES STORAGE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILES_FILE = os.path.join(BASE_DIR, "user_profiles.json")

def load_user_profiles():
    if os.path.exists(USER_PROFILES_FILE):
        try:
            with open(USER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    # Ensure each profile has required fields including password
                    for u, stats in data.items():
                        if "password" not in stats:
                            stats["password"] = "rupa" if u.lower() == "rupa" else ""
                    return data
        except Exception:
            pass
    initial_profiles = {
        "Rupa": {
            "password": "rupa",
            "score": 0,
            "streak": 0,
            "quizzes": 0,
            "mastered": [],
            "mistakes": []
        }
    }
    save_user_profiles(initial_profiles)
    return initial_profiles

def save_user_profiles(profiles):
    try:
        with open(USER_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2)
    except Exception:
        pass


# Dictionary of signing tips and descriptions
LESSON_INSTRUCTIONS = {
    "HELLO": {
        "category": "Greetings & Basics",
        "hand": "Right Hand",
        "description": "Raise your right hand to your right temple with open palm facing forward, then move your hand outwards in a friendly greeting gesture.",
        "tips": "• Keep fingers flat together.\n• Smooth outward saluting movement."
    },
    "PLEASE": {
        "category": "Greetings & Basics",
        "hand": "Right Hand",
        "description": "Place your open right palm flat against the center of your chest and move it in a polite circular motion.",
        "tips": "• Move clockwise on chest.\n• Gentle pressing motion."
    },
    "HELP": {
        "category": "Greetings & Basics",
        "hand": "Both Hands",
        "description": "Place your right closed fist (with thumb pointing up) on top of your flat left palm, then lift both hands upward together.",
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
    def __init__(self, parent, model, label_map):
        super().__init__(parent, fg_color="#080D1A", corner_radius=0)
        self.model = model
        self.label_map = label_map
        self.avatar_engine = Desktop3DAvatarEngineV2()
        
        # User Profiles
        self.profiles = load_user_profiles()
        if not self.profiles:
            self.profiles = {"Rupa": {"password": "rupa", "score": 0, "streak": 0, "quizzes": 0, "mastered": [], "mistakes": []}}
            save_user_profiles(self.profiles)
        self.active_user = "Rupa" if "Rupa" in self.profiles else list(self.profiles.keys())[0]
        self.current_tab = 1

        # Lesson state
        self.is_practice_running = False
        self.is_demo_playing = False
        self.demo_mode = "video"
        self.practice_queue = queue.Queue(maxsize=1)
        self.selected_target_gloss = "DOCTOR"

        # Quiz Engine & Customization State
        self.quiz_state = "setup"  # "setup", "in_progress", "results"
        self.quiz_submode = "video_trivia"  # "video_trivia" or "live_challenge"
        self.quiz_selected_category = "All 300 Signs"
        self.quiz_question_count = 5
        self.quiz_current_index = 0
        self.quiz_questions_list = []
        self.quiz_session_correct = 0
        self.quiz_session_mistakes = []
        self.quiz_session_mastered = []

        # Mode 1 Trivia state
        self.trivia_target_word = "HELLO"
        self.trivia_options = []
        self.trivia_answered = False
        self.is_quiz_video_playing = False

        # Mode 2 Live Camera state
        self.is_quiz_camera_running = False
        self.quiz_queue = queue.Queue(maxsize=1)

        self.setup_ui()

    def setup_ui(self):
        # Top Navigation Sub-Tab Bar
        sub_tab_frame = ctk.CTkFrame(self, fg_color="#0F172A", height=52, corner_radius=12, border_width=1, border_color="#1E293B")
        sub_tab_frame.pack(fill="x", padx=20, pady=(15, 10))
        
        self.btn_tab1 = ctk.CTkButton(
            sub_tab_frame, text="  📚 1. Lesson Curriculum  ", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", width=190, height=38, corner_radius=8,
            command=self.show_tab1
        )
        self.btn_tab1.pack(side="left", padx=10, pady=7)

        self.btn_tab2 = ctk.CTkButton(
            sub_tab_frame, text="  🎥 2. Live AI Practice  ", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=190, height=38, corner_radius=8,
            command=self.show_tab2
        )
        self.btn_tab2.pack(side="left", padx=10, pady=7)

        self.btn_tab3 = ctk.CTkButton(
            sub_tab_frame, text="  🏆 3. Interactive Skill Quiz  ", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=190, height=38, corner_radius=8,
            command=self.show_tab3
        )
        self.btn_tab3.pack(side="left", padx=10, pady=7)

        # Active User Profile Badge on the right
        profile_frame = ctk.CTkFrame(sub_tab_frame, fg_color="#16223F", corner_radius=8)
        profile_frame.pack(side="right", padx=12, pady=7)
        ctk.CTkLabel(profile_frame, text="👤", font=ctk.CTkFont(size=12), text_color="#38BDF8").pack(side="left", padx=(8, 2))
        self.lbl_active_user = ctk.CTkLabel(profile_frame, text=self.active_user, font=ctk.CTkFont(size=11, weight="bold"), text_color="#F8FAFC")
        self.lbl_active_user.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            profile_frame, text="Switch", width=52, height=24, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", corner_radius=6, command=self.open_switch_user_dialog
        ).pack(side="left", padx=(2, 4), pady=2)

        ctk.CTkButton(
            profile_frame, text="+ New", width=48, height=24, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", corner_radius=6, command=self.open_create_user_dialog
        ).pack(side="left", padx=(0, 6), pady=2)

        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.show_tab1()

    def clear_content(self):
        if self.is_practice_running:
            self.stop_practice()
        self.is_demo_playing = False
        self.is_quiz_video_playing = False
        self.is_quiz_camera_running = False
        if hasattr(self, 'content_container') and self.content_container.winfo_exists():
            for w in self.content_container.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

    # -------------------------------------------------------------
    # TAB 1: GUIDED LESSON CURRICULUM + 300-WORD INSTANT SEARCH
    # -------------------------------------------------------------
    def show_tab1(self):
        self.current_tab = 1
        self.clear_content()
        self.btn_tab1.configure(fg_color="#0284C7")
        self.btn_tab2.configure(fg_color="#1E293B")
        self.btn_tab3.configure(fg_color="#1E293B")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)

        # 1. Top Search Header Banner
        header = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=14, border_width=1, border_color="#1E293B")
        header.pack(fill="x", pady=(0, 14))

        h_row = ctk.CTkFrame(header, fg_color="transparent")
        h_row.pack(fill="x", padx=20, pady=(14, 4))
        ctk.CTkLabel(h_row, text="🎓 LearnSign Curriculum & General Lexicon", font=ctk.CTkFont(size=18, weight="bold"), text_color="#00F0FF").pack(side="left")
        
        ctk.CTkLabel(header, text="Explore curated sign tracks below, or search for any of the 300 WLASL vocabulary signs to watch real signer video demonstrations.", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 12))

        # 2. INSTANT SEARCH BAR ROW (Fixes Faculty Question #2!)
        search_card = ctk.CTkFrame(page, fg_color="#16223F", corner_radius=12, border_width=1, border_color="#1F3158")
        search_card.pack(fill="x", pady=(0, 14))

        s_inner = ctk.CTkFrame(search_card, fg_color="transparent")
        s_inner.pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(s_inner, text="🔍 Search & Learn Any Sign:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left", padx=(0, 10))

        all_words = sorted(list(self.label_map.values())) if self.label_map else sorted(list(LESSON_INSTRUCTIONS.keys()))
        self.cbo_search = ctk.CTkComboBox(s_inner, values=all_words, width=280, font=ctk.CTkFont(size=12, weight="bold"), height=36)
        self.cbo_search.set("DOCTOR")
        self.cbo_search.pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            s_inner, text="📖 Open Lesson & Video Demo ➔", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", height=36, corner_radius=8,
            command=lambda: self.show_interactive_lesson_study_page(self.cbo_search.get().strip().upper())
        ).pack(side="left")

        # 3. CURRICULUM GRID (4 Thematic Tracks)
        grid_frame = ctk.CTkFrame(page, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)
        grid_frame.grid_columnconfigure((0, 1), weight=1)
        grid_frame.grid_rowconfigure((0, 1), weight=1)

        lessons = [
            ("👋 Greetings & Basics", ["HELLO", "PLEASE", "HELP", "THANK YOU"], "#0284C7"),
            ("🏥 Medical & Healthcare", ["DOCTOR", "HOSPITAL", "PAIN", "SICK"], "#10B981"),
            ("💻 Daily Vocabulary", ["COMPUTER", "BOOK", "DRINK", "CLOTHES"], "#F59E0B"),
            ("⚡ Actions & Expressions", ["GO", "BEFORE", "CHAIR", "COME"], "#8B5CF6")
        ]

        for idx, (cat_title, words, accent_color) in enumerate(lessons):
            r, c = idx // 2, idx % 2
            card = ctk.CTkFrame(grid_frame, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
            
            top_bar = ctk.CTkFrame(card, fg_color="transparent")
            top_bar.pack(fill="x", padx=16, pady=(14, 4))
            ctk.CTkLabel(top_bar, text=cat_title, font=ctk.CTkFont(size=15, weight="bold"), text_color=accent_color).pack(side="left")
            
            words_frame = ctk.CTkFrame(card, fg_color="transparent")
            words_frame.pack(fill="both", expand=True, padx=16, pady=(4, 12))
            for w in words:
                btn = ctk.CTkButton(
                    words_frame, text=f"• Learn Sign:  '{w}'", font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color="#16223F", hover_color=accent_color, height=34, corner_radius=8, anchor="w",
                    command=lambda word=w: self.show_interactive_lesson_study_page(word)
                )
                btn.pack(fill="x", pady=3)

    # -------------------------------------------------------------
    # DEDICATED LESSON STUDY: REAL HUMAN VIDEO + 3D AVATAR TOGGLE
    # -------------------------------------------------------------
    def show_interactive_lesson_study_page(self, word):
        self.clear_content()
        self.selected_target_gloss = word
        
        info = LESSON_INSTRUCTIONS.get(word, {
            "category": "General Lexicon",
            "hand": "Right/Both Hands",
            "description": f"Perform the natural sign gesture for '{word}'. Observe the signer video on the left.",
            "tips": "• Watch hand orientation.\n• Follow the clear motion trajectory."
        })

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=2)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        # LEFT CARD: VIDEO / AVATAR DEMO
        demo_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        demo_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        title_bar = ctk.CTkFrame(demo_card, fg_color="transparent")
        title_bar.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(title_bar, text=f"Sign Demonstration: '{word}'", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00F0FF").pack(side="left")

        self.btn_toggle_demo_mode = ctk.CTkButton(
            title_bar, text="Switch to 🤖 3D Avatar Mesh", font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=32, corner_radius=8,
            command=self.toggle_demo_view_mode
        )
        self.btn_toggle_demo_mode.pack(side="right")

        self.lbl_demo_canvas = ctk.CTkLabel(demo_card, text="Loading Sign Demonstration...", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#080D1A", corner_radius=10)
        self.lbl_demo_canvas.pack(fill="both", expand=True, padx=18, pady=8)

        ctrl_bar = ctk.CTkFrame(demo_card, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=18, pady=(4, 16))

        self.lbl_mode_status = ctk.CTkLabel(ctrl_bar, text="Mode: 🎬 Authentic Human Deaf Signer Video (Dataset Reference)", font=ctk.CTkFont(size=11), text_color="#34D399")
        self.lbl_mode_status.pack(side="left")

        ctk.CTkButton(
            ctrl_bar, text="🔄 Replay Loop", font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#16223F", hover_color="#1F3158", height=30, width=110, corner_radius=6,
            command=lambda: self.start_lesson_demo(word)
        ).pack(side="right")

        # RIGHT CARD: LESSON INSTRUCTIONS & PRACTICE TRIGGER
        info_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        info_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(info_card, text=f"LESSON: '{word}'", font=ctk.CTkFont(size=22, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(20, 2))
        
        badge = ctk.CTkFrame(info_card, fg_color="#16223F", corner_radius=6)
        badge.pack(anchor="w", padx=20, pady=(0, 14))
        ctk.CTkLabel(badge, text=f" Track: {info['category']} | Hand: {info['hand']} ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(padx=8, pady=4)

        desc_box = ctk.CTkFrame(info_card, fg_color="#16223F", corner_radius=10)
        desc_box.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(desc_box, text="📖 HOW TO PERFORM THIS SIGN", font=ctk.CTkFont(size=11, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=14, pady=(10, 4))
        ctk.CTkLabel(desc_box, text=info['description'], font=ctk.CTkFont(size=12), text_color="#F8FAFC", wraplength=270, justify="left").pack(anchor="w", padx=14, pady=(0, 10))

        tips_box = ctk.CTkFrame(info_card, fg_color="#16223F", corner_radius=10)
        tips_box.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(tips_box, text="💡 KEY SIGNING TIPS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FACC15").pack(anchor="w", padx=14, pady=(10, 4))
        ctk.CTkLabel(tips_box, text=info['tips'], font=ctk.CTkFont(size=12), text_color="#CBD5E1", justify="left").pack(anchor="w", padx=14, pady=(0, 10))

        ctk.CTkButton(
            info_card, text=f"Ready to Practice '{word}' on Camera ➔", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=lambda: self.launch_practice_for_word(word)
        ).pack(fill="x", padx=20, pady=(20, 8))

        ctk.CTkButton(
            info_card, text="↩ Back to Curriculum Library", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=36, corner_radius=8,
            command=self.show_tab1
        ).pack(fill="x", padx=20, pady=6)

        self.demo_mode = "video"
        self.start_lesson_demo(word)

    def toggle_demo_view_mode(self):
        if self.demo_mode == "video":
            self.demo_mode = "avatar"
            self.btn_toggle_demo_mode.configure(text="Switch to 🎬 Real Human Video")
            self.lbl_mode_status.configure(text="Mode: 🤖 AI 3D Mesh Avatar (Computer Vision Landmark Skeleton)")
        else:
            self.demo_mode = "video"
            self.btn_toggle_demo_mode.configure(text="Switch to 🤖 3D Avatar Mesh")
            self.lbl_mode_status.configure(text="Mode: 🎬 Authentic Human Deaf Signer Video (Dataset Reference)")
        self.start_lesson_demo(self.selected_target_gloss)

    def start_lesson_demo(self, word):
        self.is_demo_playing = False
        time.sleep(0.05)
        self.is_demo_playing = True
        threading.Thread(target=self._lesson_demo_worker, args=(word,), daemon=True).start()

    def _lesson_demo_worker(self, word):
        v_path = self.avatar_engine.gloss_map.get(word)
        if not v_path or not os.path.exists(v_path):
            if hasattr(self, 'lbl_demo_canvas') and self.lbl_demo_canvas.winfo_exists():
                self.lbl_demo_canvas.configure(text=f"No dataset video found for '{word}'.", image="")
            return

        extractor = DesktopMediaPipeExtractor()

        if self.demo_mode == "video":
            while self.is_demo_playing and hasattr(self, 'lbl_demo_canvas') and self.lbl_demo_canvas.winfo_exists():
                cap = cv2.VideoCapture(v_path)
                while cap.isOpened() and self.is_demo_playing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb).resize((600, 450))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
                    
                    if hasattr(self, 'lbl_demo_canvas') and self.lbl_demo_canvas.winfo_exists():
                        self.lbl_demo_canvas.configure(image=ctk_img, text="")
                    time.sleep(0.035)
                cap.release()
                time.sleep(0.4)
        else:
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
                        skel_img = self.avatar_engine.draw_skeleton(pts, word, idx+1, total_f)
                        img = Image.fromarray(skel_img).resize((600, 450))
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
    # TAB 2: LIVE AI PRACTICE (REAL-TIME WEBCAM POSE SCORER)
    # -------------------------------------------------------------
    def show_tab2(self):
        self.current_tab = 2
        self.clear_content()
        self.btn_tab1.configure(fg_color="#1E293B")
        self.btn_tab2.configure(fg_color="#0284C7")
        self.btn_tab3.configure(fg_color="#1E293B")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=2)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        video_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        
        ctk.CTkLabel(video_card, text=f"Live AI Practice Arena: '{self.selected_target_gloss}'", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.practice_display = ctk.CTkLabel(video_card, text="Camera Off. Click 'Start Live AI Practice' Below.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#080D1A", corner_radius=10)
        self.practice_display.pack(fill="both", expand=True, padx=20, pady=10)

        self.btn_toggle_practice = ctk.CTkButton(
            video_card, text="Start Live AI Practice", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=self.toggle_practice
        )
        self.btn_toggle_practice.pack(fill="x", padx=20, pady=(5, 18))

        # Feedback & Score Card
        eval_card = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        eval_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(eval_card, text="Live AI Pose Alignment", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(20, 10))

        card_select = ctk.CTkFrame(eval_card, fg_color="#16223F", corner_radius=10)
        card_select.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_select, text="PRACTICE TARGET SIGN", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=14, pady=(10, 2))
        
        all_words = sorted(list(self.label_map.values())) if self.label_map else sorted(list(LESSON_INSTRUCTIONS.keys()))
        self.cbo_words = ctk.CTkComboBox(card_select, values=all_words, font=ctk.CTkFont(size=13, weight="bold"), command=self.on_gloss_selected)
        self.cbo_words.set(self.selected_target_gloss)
        self.cbo_words.pack(fill="x", padx=14, pady=(0, 12))

        card_score = ctk.CTkFrame(eval_card, fg_color="#16223F", corner_radius=10)
        card_score.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_score, text="ACCURACY MATCH (COSINE SIMILARITY)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=14, pady=(10, 2))
        self.lbl_score_val = ctk.CTkLabel(card_score, text="0.0%", font=ctk.CTkFont(size=22, weight="bold"), text_color="#38BDF8")
        self.lbl_score_val.pack(anchor="w", padx=14, pady=(0, 4))
        self.prog_score = ctk.CTkProgressBar(card_score, height=10, progress_color="#38BDF8")
        self.prog_score.set(0.0)
        self.prog_score.pack(fill="x", padx=14, pady=(0, 12))

        card_fb = ctk.CTkFrame(eval_card, fg_color="#16223F", corner_radius=10)
        card_fb.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(card_fb, text="REAL-TIME EVALUATION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=14, pady=(10, 2))
        self.lbl_fb_badge = ctk.CTkLabel(card_fb, text="WAITING FOR WEBCAM...", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0")
        self.lbl_fb_badge.pack(anchor="w", padx=14, pady=(0, 12))

    def on_gloss_selected(self, val):
        self.selected_target_gloss = val

    def toggle_practice(self):
        if not self.is_practice_running:
            self.is_practice_running = True
            self.btn_toggle_practice.configure(text="Stop Live AI Practice", fg_color="#DC2626", hover_color="#B91C1C")
            threading.Thread(target=self.practice_worker_thread, daemon=True).start()
            self.check_practice_queue()
        else:
            self.stop_practice()

    def stop_practice(self):
        self.is_practice_running = False
        if hasattr(self, 'btn_toggle_practice'):
            self.btn_toggle_practice.configure(text="Start Live AI Practice", fg_color="#10B981", hover_color="#059669")

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
            img = Image.fromarray(rgb_frame).resize((600, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            
            try:
                self.practice_queue.put_nowait((ctk_img, score_pct, fb_text, fb_color))
            except queue.Full:
                pass
                
            time.sleep(0.01)
            
        cap.release()
        extractor.hands.close()

    # -------------------------------------------------------------
    # TAB 3: QUIZ CONTROLLER (SETUP -> PLAYING -> RESULTS)
    # -------------------------------------------------------------
    def show_tab3(self):
        self.current_tab = 3
        self.clear_content()
        self.btn_tab1.configure(fg_color="#1E293B")
        self.btn_tab2.configure(fg_color="#1E293B")
        self.btn_tab3.configure(fg_color="#0284C7")

        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.pack(fill="both", expand=True)

        self.quiz_container = page
        
        if self.quiz_state == "setup":
            self.render_quiz_setup_screen()
        elif self.quiz_state == "in_progress":
            self.render_quiz_active_screen()
        elif self.quiz_state == "results":
            self.render_quiz_results_screen()

    def ensure_quiz_container(self):
        if not hasattr(self, 'quiz_container') or not self.quiz_container.winfo_exists():
            self.quiz_container = ctk.CTkFrame(self.content_container, fg_color="transparent")
            self.quiz_container.pack(fill="both", expand=True)
        return self.quiz_container

    # -------------------------------------------------------------
    # QUIZ SUB-SCREEN 1: SETUP & CUSTOMIZATION SCREEN
    # -------------------------------------------------------------
    def render_quiz_setup_screen(self):
        self.ensure_quiz_container()
        for w in self.quiz_container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        # Header Card
        header = ctk.CTkFrame(self.quiz_container, fg_color="#0F172A", corner_radius=14, border_width=1, border_color="#1E293B")
        header.pack(fill="x", pady=(0, 14))

        ctk.CTkLabel(header, text="🏆 Quiz Setup & Learner Profile Settings", font=ctk.CTkFont(size=18, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(14, 4))
        ctk.CTkLabel(header, text="Customize your practice session before starting. Select your profile, category, quiz format, and question count.", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 14))

        # Main Grid (Left Setup Card, Right User Stats Card)
        setup_grid = ctk.CTkFrame(self.quiz_container, fg_color="transparent")
        setup_grid.pack(fill="both", expand=True)
        setup_grid.grid_columnconfigure(0, weight=3)
        setup_grid.grid_columnconfigure(1, weight=2)
        setup_grid.grid_rowconfigure(0, weight=1)

        # LEFT CARD: CONFIGURATION CONTROLS
        cfg_card = ctk.CTkFrame(setup_grid, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        cfg_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        ctk.CTkLabel(cfg_card, text="⚙️ Custom Practice Configuration", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(16, 12))

        # 1. Profile Selection Row
        row_prof = ctk.CTkFrame(cfg_card, fg_color="#16223F", corner_radius=10)
        row_prof.pack(fill="x", padx=20, pady=6)
        
        left_prof = ctk.CTkFrame(row_prof, fg_color="transparent")
        left_prof.pack(side="left", padx=12, pady=8)
        ctk.CTkLabel(left_prof, text="Active Learner Profile:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left")
        ctk.CTkLabel(left_prof, text=f"👤 {self.active_user}", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC").pack(side="left", padx=(8, 0))

        btn_new_user = ctk.CTkButton(
            row_prof, text="➕ Create New User", width=130, height=30, font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", command=self.open_create_user_dialog
        )
        btn_new_user.pack(side="right", padx=10, pady=8)

        btn_switch_user = ctk.CTkButton(
            row_prof, text="🔄 Switch Profile", width=120, height=30, font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#334155", hover_color="#475569", command=self.open_switch_user_dialog
        )
        btn_switch_user.pack(side="right", padx=(4, 4), pady=8)

        # 2. Category Selection Row
        row_cat = ctk.CTkFrame(cfg_card, fg_color="#16223F", corner_radius=10)
        row_cat.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(row_cat, text="Target Practice Category:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left", padx=12, pady=10)
        
        categories = ["All 300 Signs", "Greetings & Basics", "Medical & Healthcare", "Daily Vocabulary", "Actions & Expressions"]
        self.cbo_category = ctk.CTkComboBox(row_cat, values=categories, font=ctk.CTkFont(size=12), width=170)
        self.cbo_category.set(self.quiz_selected_category)
        self.cbo_category.pack(side="right", padx=12, pady=10)

        # 3. Question Count Row
        row_cnt = ctk.CTkFrame(cfg_card, fg_color="#16223F", corner_radius=10)
        row_cnt.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(row_cnt, text="Number of Questions:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left", padx=12, pady=10)
        
        self.cbo_count = ctk.CTkComboBox(row_cnt, values=["5 Questions", "10 Questions", "15 Questions"], font=ctk.CTkFont(size=12), width=170)
        self.cbo_count.set(f"{self.quiz_question_count} Questions")
        self.cbo_count.pack(side="right", padx=12, pady=10)

        # 4. Quiz Mode Selector
        row_mode = ctk.CTkFrame(cfg_card, fg_color="#16223F", corner_radius=10)
        row_mode.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(row_mode, text="Quiz Format Mode:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left", padx=12, pady=10)
        
        self.cbo_mode = ctk.CTkComboBox(row_mode, values=["🎬 Mode 1: Guess The Sign (Video Quiz)", "🎥 Mode 2: Live Camera AI Action Challenge"], font=ctk.CTkFont(size=12), width=240)
        self.cbo_mode.set("🎬 Mode 1: Guess The Sign (Video Quiz)" if self.quiz_submode == "video_trivia" else "🎥 Mode 2: Live Camera AI Action Challenge")
        self.cbo_mode.pack(side="right", padx=12, pady=10)

        # Start Quiz Button
        ctk.CTkButton(
            cfg_card, text="🚀 Start Practice Quiz Session ➔", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=46, corner_radius=10,
            command=self.start_custom_quiz_session
        ).pack(fill="x", padx=20, pady=(18, 16))

        # RIGHT CARD: USER STATS & MISTAKES LOG
        stats_card = ctk.CTkFrame(setup_grid, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        stats_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(stats_card, text=f"📊 Learner Profile: {self.active_user}", font=ctk.CTkFont(size=15, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=18, pady=(16, 4))
        
        user_data = self.profiles.get(self.active_user, {"score": 0, "streak": 0, "quizzes": 0, "mastered": [], "mistakes": []})
        
        stats_rows = [
            ("⭐ Total Points Earned:", f"{user_data.get('score', 0)} pts"),
            ("🔥 Best Streak Counter:", f"{user_data.get('streak', 0)} streak"),
            ("📝 Total Quizzes Taken:", f"{user_data.get('quizzes', 0)} quizzes"),
            ("🎯 Signs Mastered:", f"{len(user_data.get('mastered', []))} signs"),
        ]
        for lbl, val in stats_rows:
            r = ctk.CTkFrame(stats_card, fg_color="#16223F", corner_radius=8)
            r.pack(fill="x", padx=18, pady=4)
            ctk.CTkLabel(r, text=lbl, font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(side="left", padx=10, pady=6)
            ctk.CTkLabel(r, text=val, font=ctk.CTkFont(size=12, weight="bold"), text_color="#34D399").pack(side="right", padx=10, pady=6)

        # Recent Mistakes to Practice
        ctk.CTkLabel(stats_card, text="⚠️ Recent Signs to Review:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#FACC15").pack(anchor="w", padx=18, pady=(12, 4))
        
        mistakes = user_data.get("mistakes", [])[-4:]
        if mistakes:
            m_frame = ctk.CTkFrame(stats_card, fg_color="transparent")
            m_frame.pack(fill="x", padx=18, pady=4)
            for m_word in mistakes:
                btn_m = ctk.CTkButton(
                    m_frame, text=f"• Re-Study '{m_word}' ➔", font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#1E293B", hover_color="#DC2626", height=28, corner_radius=6, anchor="w",
                    command=lambda w=m_word: self.show_interactive_lesson_study_page(w)
                )
                btn_m.pack(fill="x", pady=2)
        else:
            ctk.CTkLabel(stats_card, text="No mistakes recorded yet! Take a quiz to test your memory.", font=ctk.CTkFont(size=11), text_color="#64748B", wraplength=220).pack(anchor="w", padx=18, pady=(0, 10))

    def open_switch_user_dialog(self):
        self.profiles = load_user_profiles()
        dialog = ctk.CTkToplevel(self)
        dialog.title("Switch Learner Profile - SignVerse")
        dialog.geometry("420x410")
        dialog.resizable(False, False)
        dialog.configure(fg_color="#0B132B")
        dialog.transient(self.winfo_toplevel())
        dialog.attributes("-topmost", True)
        dialog.grab_set()

        try:
            dialog.update_idletasks()
            top = self.winfo_toplevel()
            x = top.winfo_x() + (top.winfo_width() // 2) - 210
            y = top.winfo_y() + (top.winfo_height() // 2) - 205
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        card = ctk.CTkFrame(dialog, fg_color="#0F172A", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(card, text="🔐 Switch Learner Profile", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(card, text="Select your learner profile and enter your password.", font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkLabel(card, text="Select Profile:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(4, 2))
        users_list = list(self.profiles.keys()) if self.profiles else ["Rupa"]
        cbo = ctk.CTkComboBox(card, values=users_list, font=ctk.CTkFont(size=12), height=34)
        cbo.set(self.active_user if self.active_user in users_list else users_list[0])
        cbo.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(card, text="Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(4, 2))
        txt_pwd = ctk.CTkEntry(card, placeholder_text="Enter password", show="*", font=ctk.CTkFont(size=12), height=34)
        txt_pwd.pack(fill="x", padx=20, pady=(0, 6))

        lbl_err = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color="#EF4444")
        lbl_err.pack(anchor="w", padx=20, pady=(0, 8))

        def try_login():
            target_user = cbo.get()
            pwd = txt_pwd.get().strip()
            if not target_user or target_user not in self.profiles:
                lbl_err.configure(text="❌ Selected profile not found.")
                return
            
            user_info = self.profiles[target_user]
            expected_pwd = user_info.get("password", "")
            if expected_pwd and pwd != expected_pwd:
                lbl_err.configure(text="❌ Incorrect password. Please try again.")
                return
            
            self.active_user = target_user
            self.lbl_active_user.configure(text=self.active_user)
            if hasattr(self, 'current_tab') and self.current_tab == 3 and hasattr(self, 'quiz_state') and self.quiz_state == "setup":
                self.render_quiz_setup_screen()
            dialog.destroy()

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(4, 12))

        ctk.CTkButton(
            btn_row, text="Login ➔", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", height=36,
            command=try_login
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Cancel", font=ctk.CTkFont(size=12),
            fg_color="#334155", hover_color="#475569", height=36, width=80,
            command=dialog.destroy
        ).pack(side="right")

        div = ctk.CTkFrame(card, fg_color="#1E293B", height=1)
        div.pack(fill="x", padx=20, pady=(4, 10))

        def switch_to_create():
            dialog.destroy()
            self.open_create_user_dialog()

        ctk.CTkButton(
            card, text="➕ Need a new profile? Create New User", font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#16223F", hover_color="#1E293B", text_color="#38BDF8", height=32,
            command=switch_to_create
        ).pack(fill="x", padx=20, pady=(0, 10))

        txt_pwd.bind("<Return>", lambda e: try_login())
        txt_pwd.focus()

    def open_create_user_dialog(self):
        self.profiles = load_user_profiles()
        dialog = ctk.CTkToplevel(self)
        dialog.title("Create New Learner Profile - SignVerse")
        dialog.geometry("420x470")
        dialog.resizable(False, False)
        dialog.configure(fg_color="#0B132B")
        dialog.transient(self.winfo_toplevel())
        dialog.attributes("-topmost", True)
        dialog.grab_set()

        try:
            dialog.update_idletasks()
            top = self.winfo_toplevel()
            x = top.winfo_x() + (top.winfo_width() // 2) - 210
            y = top.winfo_y() + (top.winfo_height() // 2) - 235
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        card = ctk.CTkFrame(dialog, fg_color="#0F172A", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(card, text="✨ Create New Learner Profile", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(card, text="Register a unique learner name & password to track your scores.", font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 12))

        ctk.CTkLabel(card, text="Learner Name / Username:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(2, 2))
        txt_user = ctk.CTkEntry(card, placeholder_text="e.g. Alex, Sam, Maria", font=ctk.CTkFont(size=12), height=32)
        txt_user.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkLabel(card, text="Set Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(2, 2))
        txt_pwd = ctk.CTkEntry(card, placeholder_text="Enter password", show="*", font=ctk.CTkFont(size=12), height=32)
        txt_pwd.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkLabel(card, text="Confirm Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(2, 2))
        txt_cpwd = ctk.CTkEntry(card, placeholder_text="Re-type password", show="*", font=ctk.CTkFont(size=12), height=32)
        txt_cpwd.pack(fill="x", padx=20, pady=(0, 6))

        lbl_err = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color="#EF4444")
        lbl_err.pack(anchor="w", padx=20, pady=(0, 8))

        def try_register():
            new_user = txt_user.get().strip()
            pwd = txt_pwd.get().strip()
            cpwd = txt_cpwd.get().strip()

            if not new_user:
                lbl_err.configure(text="❌ Please enter a learner name.")
                return
            if len(new_user) < 2:
                lbl_err.configure(text="❌ Name must be at least 2 characters.")
                return
            if new_user in self.profiles:
                lbl_err.configure(text="❌ A profile with this name already exists.")
                return
            if not pwd:
                lbl_err.configure(text="❌ Please set a password.")
                return
            if pwd != cpwd:
                lbl_err.configure(text="❌ Passwords do not match.")
                return

            self.profiles[new_user] = {
                "password": pwd,
                "score": 0,
                "streak": 0,
                "quizzes": 0,
                "mastered": [],
                "mistakes": []
            }
            save_user_profiles(self.profiles)

            self.active_user = new_user
            self.lbl_active_user.configure(text=self.active_user)
            if hasattr(self, 'current_tab') and self.current_tab == 3 and hasattr(self, 'quiz_state') and self.quiz_state == "setup":
                self.render_quiz_setup_screen()
            dialog.destroy()

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(4, 12))

        ctk.CTkButton(
            btn_row, text="Create Profile ➔", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=36,
            command=try_register
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Cancel", font=ctk.CTkFont(size=12),
            fg_color="#334155", hover_color="#475569", height=36, width=80,
            command=dialog.destroy
        ).pack(side="right")

        txt_cpwd.bind("<Return>", lambda e: try_register())
        txt_user.focus()

    def on_profile_switched(self, val=None):
        self.open_switch_user_dialog()

    def start_custom_quiz_session(self):
        # Read user selections
        self.quiz_selected_category = self.cbo_category.get()
        count_str = self.cbo_count.get()
        self.quiz_question_count = int(count_str.split()[0])
        mode_str = self.cbo_mode.get()
        self.quiz_submode = "video_trivia" if "Mode 1" in mode_str else "live_challenge"

        # Filter candidate words based on category
        available_glosses = [k for k in self.avatar_engine.gloss_map.keys() if os.path.exists(self.avatar_engine.gloss_map[k])]
        if not available_glosses:
            available_glosses = ["HELLO", "DOCTOR", "COMPUTER", "HELP", "BOOK", "PLEASE", "CHAIR", "GO", "DRINK"]

        if self.quiz_selected_category == "Greetings & Basics":
            cat_words = ["HELLO", "PLEASE", "HELP", "THANK YOU"]
        elif self.quiz_selected_category == "Medical & Healthcare":
            cat_words = ["DOCTOR", "HOSPITAL", "PAIN", "SICK"]
        elif self.quiz_selected_category == "Daily Vocabulary":
            cat_words = ["COMPUTER", "BOOK", "DRINK", "CLOTHES"]
        elif self.quiz_selected_category == "Actions & Expressions":
            cat_words = ["GO", "BEFORE", "CHAIR", "COME"]
        else:
            cat_words = available_glosses

        filtered = [w for w in cat_words if w in available_glosses]
        if not filtered:
            filtered = available_glosses

        # Build quiz sequence
        random.shuffle(filtered)
        while len(filtered) < self.quiz_question_count:
            filtered.extend(available_glosses)
        self.quiz_questions_list = filtered[:self.quiz_question_count]

        self.quiz_current_index = 0
        self.quiz_session_correct = 0
        self.quiz_session_mistakes = []
        self.quiz_session_mastered = []
        self.quiz_state = "in_progress"

        self.render_quiz_active_screen()

    # -------------------------------------------------------------
    # QUIZ SUB-SCREEN 2: ACTIVE QUESTION SCREEN
    # -------------------------------------------------------------
    def render_quiz_active_screen(self):
        self.ensure_quiz_container()
        for w in self.quiz_container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        if self.quiz_current_index >= len(self.quiz_questions_list):
            self.finalize_quiz_session()
            return

        target_word = self.quiz_questions_list[self.quiz_current_index]

        # Top Session Tracker Banner
        top_bar = ctk.CTkFrame(self.quiz_container, fg_color="#0F172A", corner_radius=12, border_width=1, border_color="#1E293B")
        top_bar.pack(fill="x", pady=(0, 12))

        tb_inner = ctk.CTkFrame(top_bar, fg_color="transparent")
        tb_inner.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            tb_inner, 
            text=f"Question {self.quiz_current_index + 1} of {self.quiz_question_count}  |  Category: {self.quiz_selected_category}", 
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#00F0FF"
        ).pack(side="left")

        # Exit Quiz Early button
        ctk.CTkButton(
            tb_inner, text="Quit Quiz ✕", font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1E293B", hover_color="#DC2626", height=28, width=90, corner_radius=6,
            command=self.quit_quiz_early
        ).pack(side="right")

        # Body Frame for Quiz question
        body = ctk.CTkFrame(self.quiz_container, fg_color="transparent")
        body.pack(fill="both", expand=True)

        if self.quiz_submode == "video_trivia":
            self.render_trivia_question(body, target_word)
        else:
            self.render_live_challenge_question(body, target_word)

    def quit_quiz_early(self):
        self.clear_content()
        self.quiz_state = "setup"
        self.show_tab3()

    # --- Mode 1 Question Renderer ---
    def render_trivia_question(self, body, target_word):
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.trivia_target_word = target_word
        self.trivia_answered = False

        # Build 4 options
        all_glosses = list(self.avatar_engine.gloss_map.keys()) if self.avatar_engine.gloss_map else ["HELLO", "DOCTOR", "COMPUTER", "HELP", "BOOK", "PLEASE"]
        distractors = [w for w in all_glosses if w != self.trivia_target_word]
        if len(distractors) < 3:
            distractors.extend(["HELLO", "DOCTOR", "COMPUTER", "HELP", "BOOK", "PLEASE"])
            distractors = [w for w in distractors if w != self.trivia_target_word]
        random.shuffle(distractors)
        self.trivia_options = distractors[:3] + [self.trivia_target_word]
        random.shuffle(self.trivia_options)

        # LEFT: Video Player
        vid_card = ctk.CTkFrame(body, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        vid_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        ctk.CTkLabel(vid_card, text="🎬 Watch the Signer: What Sign is Being Performed?", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(14, 6))

        self.lbl_quiz_video = ctk.CTkLabel(vid_card, text="Loading Sign Video...", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#080D1A", corner_radius=10)
        self.lbl_quiz_video.pack(fill="both", expand=True, padx=20, pady=8)

        # RIGHT: 4 Multiple Choice Options
        opt_card = ctk.CTkFrame(body, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        opt_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(opt_card, text="Select the Correct Sign:", font=ctk.CTkFont(size=15, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(18, 10))

        self.opt_buttons = []
        for opt in self.trivia_options:
            btn = ctk.CTkButton(
                opt_card, text=f"•  {opt}", font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#16223F", hover_color="#0284C7", height=42, corner_radius=10, anchor="w",
                command=lambda chosen=opt: self.check_trivia_answer_advanced(chosen)
            )
            btn.pack(fill="x", padx=20, pady=5)
            self.opt_buttons.append((btn, opt))

        self.lbl_trivia_result = ctk.CTkLabel(opt_card, text="Watch the video on the left and tap your answer!", font=ctk.CTkFont(size=12), text_color="#94A3B8", wraplength=260)
        self.lbl_trivia_result.pack(anchor="w", padx=20, pady=(12, 8))

        self.btn_next_trivia = ctk.CTkButton(
            opt_card, text="Next Question ➔" if self.quiz_current_index + 1 < self.quiz_question_count else "Finish Quiz & View Report ➔", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#334155", state="disabled", height=42, corner_radius=10,
            command=self.advance_to_next_question
        )
        self.btn_next_trivia.pack(fill="x", padx=20, pady=(10, 18), side="bottom")

        self.start_quiz_video_playback(self.trivia_target_word)

    def start_quiz_video_playback(self, word):
        self.quiz_video_session = getattr(self, 'quiz_video_session', 0) + 1
        current_session = self.quiz_video_session
        self.is_quiz_video_playing = True
        threading.Thread(target=self._quiz_video_worker, args=(word, current_session), daemon=True).start()

    def _quiz_video_worker(self, word, session_id):
        v_path = self.avatar_engine.gloss_map.get(word)
        if not v_path or not os.path.exists(v_path):
            return

        while (self.is_quiz_video_playing and 
               getattr(self, 'quiz_video_session', 0) == session_id and 
               hasattr(self, 'lbl_quiz_video') and 
               self.lbl_quiz_video.winfo_exists()):
            cap = cv2.VideoCapture(v_path)
            while (cap.isOpened() and 
                   self.is_quiz_video_playing and 
                   getattr(self, 'quiz_video_session', 0) == session_id and
                   hasattr(self, 'lbl_quiz_video') and 
                   self.lbl_quiz_video.winfo_exists()):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((600, 450))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
                
                try:
                    if (getattr(self, 'quiz_video_session', 0) == session_id and 
                        hasattr(self, 'lbl_quiz_video') and 
                        self.lbl_quiz_video.winfo_exists()):
                        self.lbl_quiz_video.configure(image=ctk_img, text="")
                except Exception:
                    break
                time.sleep(0.035)
            cap.release()
            time.sleep(0.4)

    def check_trivia_answer_advanced(self, chosen):
        if self.trivia_answered:
            return
        self.trivia_answered = True

        if chosen == self.trivia_target_word:
            self.quiz_session_correct += 1
            self.quiz_session_mastered.append(self.trivia_target_word)
            self.lbl_trivia_result.configure(text=f"🎉 CORRECT! That was '{self.trivia_target_word}'! (+10 pts)", text_color="#4ADE80")
            speak_audio("Correct! Great job!")
        else:
            self.quiz_session_mistakes.append(self.trivia_target_word)
            self.lbl_trivia_result.configure(text=f"❌ INCORRECT! The correct sign was '{self.trivia_target_word}'.", text_color="#F87171")
            speak_audio(f"That was {self.trivia_target_word}")

        for btn, opt in self.opt_buttons:
            if opt == self.trivia_target_word:
                btn.configure(fg_color="#065F46", text_color="#34D399")
            elif opt == chosen:
                btn.configure(fg_color="#7F1D1D", text_color="#F87171")

        self.btn_next_trivia.configure(state="normal", fg_color="#10B981")

    # --- Mode 2 Live Camera Question Renderer (REAL AI EVALUATION) ---
    def render_live_challenge_question(self, body, target_word):
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.quiz_challenge_word = target_word

        # LEFT: LIVE WEBCAM ARENA
        cam_card = ctk.CTkFrame(body, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        ctk.CTkLabel(cam_card, text=f"🎥 Live AI Camera Challenge: Show Sign for '{self.quiz_challenge_word}'", font=ctk.CTkFont(size=15, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=20, pady=(14, 6))

        self.lbl_live_quiz_cam = ctk.CTkLabel(cam_card, text="Click 'Start 6s Camera Challenge' Below When Ready.", font=ctk.CTkFont(size=13), text_color="#64748B", fg_color="#080D1A", corner_radius=10)
        self.lbl_live_quiz_cam.pack(fill="both", expand=True, padx=20, pady=8)

        self.btn_start_challenge = ctk.CTkButton(
            cam_card, text="⚡ Start 6s Camera Challenge", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=self.start_live_quiz_challenge_real
        )
        self.btn_start_challenge.pack(fill="x", padx=20, pady=(4, 18))

        # RIGHT: TARGET PROMPT & REAL-TIME VERDICT
        prompt_card = ctk.CTkFrame(body, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        prompt_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(prompt_card, text="TARGET SIGN TO PERFORM", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(18, 2))
        ctk.CTkLabel(prompt_card, text=f"'{self.quiz_challenge_word}'", font=ctk.CTkFont(size=26, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=20, pady=(0, 12))

        # Countdown Timer Box
        timer_box = ctk.CTkFrame(prompt_card, fg_color="#16223F", corner_radius=10)
        timer_box.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(timer_box, text="COUNTDOWN TIMER", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=14, pady=(8, 2))
        self.lbl_timer_val = ctk.CTkLabel(timer_box, text="6.0s", font=ctk.CTkFont(size=24, weight="bold"), text_color="#FACC15")
        self.lbl_timer_val.pack(anchor="w", padx=14, pady=(0, 8))

        # Real-Time AI Model Verdict
        verdict_box = ctk.CTkFrame(prompt_card, fg_color="#16223F", corner_radius=10)
        verdict_box.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(verdict_box, text="AI CLASSIFIER DETECTION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=14, pady=(8, 2))
        self.lbl_challenge_verdict = ctk.CTkLabel(verdict_box, text="READY. PRESS START.", font=ctk.CTkFont(size=13, weight="bold"), text_color="#CBD5E1")
        self.lbl_challenge_verdict.pack(anchor="w", padx=14, pady=(0, 8))

        self.btn_next_live = ctk.CTkButton(
            prompt_card, text="Next Question ➔" if self.quiz_current_index + 1 < self.quiz_question_count else "Finish Quiz & View Report ➔", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#334155", state="disabled", height=42, corner_radius=10,
            command=self.advance_to_next_question
        )
        self.btn_next_live.pack(fill="x", padx=20, pady=(10, 18), side="bottom")

    def start_live_quiz_challenge_real(self):
        if self.is_quiz_camera_running:
            return
        self.is_quiz_camera_running = True
        self.btn_start_challenge.configure(state="disabled", fg_color="#334155")
        threading.Thread(target=self._live_quiz_worker_real_ai, daemon=True).start()
        self.check_quiz_queue_real()

    def check_quiz_queue_real(self):
        if self.is_quiz_camera_running:
            try:
                frame_img, time_left, verdict, color = self.quiz_queue.get_nowait()
                if hasattr(self, 'lbl_live_quiz_cam') and self.lbl_live_quiz_cam.winfo_exists():
                    self.lbl_live_quiz_cam.configure(image=frame_img, text="")
                    self.lbl_timer_val.configure(text=f"{time_left:.1f}s")
                    self.lbl_challenge_verdict.configure(text=verdict, text_color=color)
            except queue.Empty:
                pass
            self.after(15, self.check_quiz_queue_real)

    # REAL NEURAL NETWORK EVALUATION (Fixes Bug #1!)
    def _live_quiz_worker_real_ai(self):
        cap = cv2.VideoCapture(0)
        extractor = DesktopMediaPipeExtractor()
        sequence_buffer = []
        start_time = time.time()
        challenge_duration = 6.0
        success = False
        
        speak_audio(f"Show the sign for {self.quiz_challenge_word}!")
        
        while time.time() - start_time < challenge_duration and cap.isOpened() and self.is_quiz_camera_running:
            elapsed = time.time() - start_time
            time_left = max(0.0, challenge_duration - elapsed)
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            feats, annotated_frame, detected_count = extractor.extract_landmarks(frame)
            sequence_buffer.append(feats)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)

            verdict = "POSITION HANDS & SIGN..."
            color = "#FACC15"
            
            # REAL INFERENCE CLASSIFICATION
            if len(sequence_buffer) == 30 and detected_count > 0 and self.model:
                seq_input = np.array([sequence_buffer], dtype=np.float32)
                probs = self.model.predict(seq_input, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                pred_word = self.label_map.get(pred_idx, "Unknown")
                
                verdict = f"Detected: {pred_word} ({conf*100:.0f}%)"
                color = "#38BDF8"

                # STRICT CRITERIA: Prediction must match target word with >= 45% confidence!
                if pred_word.upper() == self.quiz_challenge_word.upper() and conf >= 0.45:
                    success = True
                    break

            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((600, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            
            try:
                self.quiz_queue.put_nowait((ctk_img, time_left, verdict, color))
            except queue.Full:
                pass
                
            time.sleep(0.02)
            
        cap.release()
        extractor.hands.close()
        self.is_quiz_camera_running = False

        if success:
            self.quiz_session_correct += 1
            self.quiz_session_mastered.append(self.quiz_challenge_word)
            if hasattr(self, 'lbl_challenge_verdict') and self.lbl_challenge_verdict.winfo_exists():
                self.lbl_challenge_verdict.configure(text=f"🎉 CORRECT SIGN: {self.quiz_challenge_word}! (+15 pts)", text_color="#4ADE80")
                self.lbl_timer_val.configure(text="0.0s")
            speak_audio("Awesome! Correct sign recognized!")
        else:
            self.quiz_session_mistakes.append(self.quiz_challenge_word)
            if hasattr(self, 'lbl_challenge_verdict') and self.lbl_challenge_verdict.winfo_exists():
                self.lbl_challenge_verdict.configure(text="❌ TIME'S UP! Sign not matched.", text_color="#F87171")
            speak_audio("Time is up! Did not match target sign.")

        if hasattr(self, 'btn_next_live') and self.btn_next_live.winfo_exists():
            self.btn_next_live.configure(state="normal", fg_color="#10B981")

    def advance_to_next_question(self):
        # Stop previous question's video/camera threads without clearing the container
        self.is_quiz_video_playing = False
        self.is_quiz_camera_running = False
        
        self.quiz_current_index += 1
        if self.quiz_current_index < self.quiz_question_count:
            self.render_quiz_active_screen()
        else:
            self.finalize_quiz_session()

    # -------------------------------------------------------------
    # QUIZ SUB-SCREEN 3: REPORT & MISTAKES REVIEW SCREEN
    # -------------------------------------------------------------
    def finalize_quiz_session(self):
        self.quiz_state = "results"
        
        # Save results to user profile
        user_data = self.profiles.setdefault(self.active_user, {"score": 0, "streak": 0, "quizzes": 0, "mastered": [], "mistakes": []})
        
        earned_pts = self.quiz_session_correct * 10
        user_data["score"] = user_data.get("score", 0) + earned_pts
        user_data["quizzes"] = user_data.get("quizzes", 0) + 1
        
        if self.quiz_session_correct == self.quiz_question_count:
            user_data["streak"] = user_data.get("streak", 0) + 1
        else:
            user_data["streak"] = 0

        # Update mastered and mistakes lists
        cur_mastered = set(user_data.get("mastered", []))
        cur_mastered.update(self.quiz_session_mastered)
        user_data["mastered"] = list(cur_mastered)

        cur_mistakes = user_data.get("mistakes", [])
        for m in self.quiz_session_mistakes:
            if m not in cur_mistakes:
                cur_mistakes.append(m)
        user_data["mistakes"] = cur_mistakes[-10:] # keep last 10

        save_user_profiles(self.profiles)
        self.render_quiz_results_screen()

    def render_quiz_results_screen(self):
        self.ensure_quiz_container()
        for w in self.quiz_container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        page = ctk.CTkFrame(self.quiz_container, fg_color="transparent")
        page.pack(fill="both", expand=True)

        # Header Banner
        header = ctk.CTkFrame(page, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        header.pack(fill="x", pady=(0, 16))

        pct = (self.quiz_session_correct / self.quiz_question_count) * 100.0 if self.quiz_question_count > 0 else 0
        h_color = "#34D399" if pct >= 70.0 else "#FACC15" if pct >= 50.0 else "#F87171"

        ctk.CTkLabel(header, text="🏆 Quiz Session Completed!", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00F0FF").pack(anchor="w", padx=25, pady=(16, 2))
        ctk.CTkLabel(header, text=f"Performance Report for Learner: {self.active_user}", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=25, pady=(0, 16))

        # Results Grid (Left Score, Right Mistakes to Review)
        grid = ctk.CTkFrame(page, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.grid_columnconfigure((0, 1), weight=1)
        grid.grid_rowconfigure(0, weight=1)

        # LEFT CARD: SCORE CARD
        score_card = ctk.CTkFrame(grid, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        score_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(score_card, text="FINAL SCORE SUMMARY", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(18, 4))
        ctk.CTkLabel(score_card, text=f"{self.quiz_session_correct} / {self.quiz_question_count} Correct ({pct:.0f}%)", font=ctk.CTkFont(size=24, weight="bold"), text_color=h_color).pack(anchor="w", padx=20, pady=(0, 12))

        prog_bar = ctk.CTkProgressBar(score_card, height=12, progress_color=h_color)
        prog_bar.set(pct / 100.0)
        prog_bar.pack(fill="x", padx=20, pady=(0, 16))

        stats_rows = [
            ("⭐ Points Earned:", f"+{self.quiz_session_correct * 10} Points"),
            ("🎯 Mastered in this Session:", f"{len(self.quiz_session_mastered)} signs"),
            ("⚠️ Missed Questions:", f"{len(self.quiz_session_mistakes)} signs"),
            ("🔥 Cumulative Total Score:", f"{self.profiles[self.active_user].get('score', 0)} pts")
        ]
        for l, v in stats_rows:
            r = ctk.CTkFrame(score_card, fg_color="#16223F", corner_radius=8)
            r.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(r, text=l, font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(side="left", padx=12, pady=6)
            ctk.CTkLabel(r, text=v, font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="right", padx=12, pady=6)

        ctk.CTkButton(
            score_card, text="🔄 Take Another Custom Quiz ➔", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", height=42, corner_radius=10,
            command=self.reset_to_quiz_setup
        ).pack(fill="x", padx=20, pady=(18, 16), side="bottom")

        # RIGHT CARD: MISTAKES REVIEW LIST WITH 1-CLICK STUDY
        mistakes_card = ctk.CTkFrame(grid, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        mistakes_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        ctk.CTkLabel(mistakes_card, text="📖 Review Your Mistakes & Learn", font=ctk.CTkFont(size=15, weight="bold"), text_color="#FACC15").pack(anchor="w", padx=20, pady=(18, 4))
        ctk.CTkLabel(mistakes_card, text="Click any missed sign below to immediately open its video lesson and instructions:", font=ctk.CTkFont(size=11), text_color="#94A3B8", wraplength=300).pack(anchor="w", padx=20, pady=(0, 12))

        if self.quiz_session_mistakes:
            scroll = ctk.CTkScrollableFrame(mistakes_card, fg_color="transparent")
            scroll.pack(fill="both", expand=True, padx=16, pady=4)

            for m_word in self.quiz_session_mistakes:
                item_card = ctk.CTkFrame(scroll, fg_color="#16223F", corner_radius=10)
                item_card.pack(fill="x", pady=4)
                
                ctk.CTkLabel(item_card, text=f"❌ '{m_word}'", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F87171").pack(side="left", padx=12, pady=8)
                
                ctk.CTkButton(
                    item_card, text="Re-Study Sign ➔", font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#0284C7", hover_color="#0369A1", height=28, width=120, corner_radius=6,
                    command=lambda w=m_word: self.show_interactive_lesson_study_page(w)
                ).pack(side="right", padx=10, pady=6)
        else:
            ctk.CTkLabel(mistakes_card, text="🎉 Outstanding Job! You got 100% of questions right!", font=ctk.CTkFont(size=13, weight="bold"), text_color="#34D399").pack(pady=40)

        ctk.CTkButton(
            mistakes_card, text="📚 Return to Curriculum Library", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=42, corner_radius=10,
            command=self.show_tab1
        ).pack(fill="x", padx=20, pady=(18, 16), side="bottom")

    def reset_to_quiz_setup(self):
        self.clear_content()
        self.quiz_state = "setup"
        self.show_tab3()
