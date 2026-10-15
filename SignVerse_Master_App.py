import os
import json
import customtkinter as ctk
import tensorflow as tf

from module1_engine import Module1View
from module2_engine import Module2View

# Modern Cyber-Slate Appearance Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignVerseMasterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SignVerse AI Platform — Next-Gen Assistive Sign Language Suite")
        self.geometry("1220x780")
        self.minsize(1080, 700)

        # Shared Model & Label Map Loading
        self.model = None
        self.label_map = {}
        self.load_model()

        # Active Module View
        self.current_view = None

        self.setup_ui_layout()
        self.show_home_hub()

    def load_model(self):
        model_path = "saved_models/best_signverse_model_300.keras"
        label_path = "preprocessed_data_300/label_map_300.json"
        if os.path.exists(model_path) and os.path.exists(label_path):
            self.model = tf.keras.models.load_model(model_path)
            with open(label_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
                self.label_map = {int(k): v for k, v in self.label_map.items()}

    def setup_ui_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # LEFT SIDEBAR NAVIGATION HUB
        # -------------------------------------------------------------
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#0B132B")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Glowing Branding Banner
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=20, pady=(25, 20), sticky="ew")
        
        ctk.CTkLabel(
            brand_frame, text="✨ SignVerse AI", 
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), 
            text_color="#00F0FF"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            brand_frame, text="Dual-Engine Assistive Platform", 
            font=ctk.CTkFont(size=11), 
            text_color="#94A3B8"
        ).pack(anchor="w", pady=(2, 0))

        # Nav Buttons with vibrant icons
        self.btn_nav_home = ctk.CTkButton(
            self.sidebar, text="  🏠  Home Dashboard", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", 
            height=44, corner_radius=10, anchor="w",
            command=self.show_home_hub
        )
        self.btn_nav_home.grid(row=2, column=0, padx=15, pady=6, sticky="ew")

        self.btn_nav_mod1 = ctk.CTkButton(
            self.sidebar, text="  💬  Module 1: Communication", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", 
            height=44, corner_radius=10, anchor="w",
            command=self.show_module1
        )
        self.btn_nav_mod1.grid(row=3, column=0, padx=15, pady=6, sticky="ew")

        self.btn_nav_mod2 = ctk.CTkButton(
            self.sidebar, text="  🎓  Module 2: LearnSign Engine", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", 
            height=44, corner_radius=10, anchor="w",
            command=self.show_module2
        )
        self.btn_nav_mod2.grid(row=4, column=0, padx=15, pady=6, sticky="ew")

        # Sidebar Footer Specs Card
        info_card = ctk.CTkFrame(self.sidebar, fg_color="#16223F", corner_radius=12, border_width=1, border_color="#1F3158")
        info_card.grid(row=7, column=0, padx=15, pady=20, sticky="ew")
        
        ctk.CTkLabel(
            info_card, text="⚙️ Platform Specs", 
            font=ctk.CTkFont(size=12, weight="bold"), 
            text_color="#38BDF8"
        ).pack(anchor="w", padx=14, pady=(10, 4))
        
        specs_text = "• WLASL-300 Benchmark\n• Conv1D + BiLSTM Model\n• 21 3D Joints (126 Dims)\n• 60+ FPS Real-Time CPU\n• Windows SAPI5 Speech"
        ctk.CTkLabel(
            info_card, text=specs_text, 
            font=ctk.CTkFont(size=11), 
            text_color="#94A3B8", justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 10))

        # -------------------------------------------------------------
        # MAIN CONTENT DISPLAY CONTAINER
        # -------------------------------------------------------------
        self.main_container = ctk.CTkFrame(self, fg_color="#080D1A", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

    def clear_container(self):
        if self.current_view:
            if hasattr(self.current_view, 'stop_camera'):
                try:
                    self.current_view.stop_camera()
                except Exception:
                    pass
            if hasattr(self.current_view, 'stop_practice'):
                try:
                    self.current_view.stop_practice()
                except Exception:
                    pass
            if hasattr(self.current_view, 'clear_content'):
                try:
                    self.current_view.clear_content()
                except Exception:
                    pass
            self.current_view = None

        for w in list(self.main_container.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass

    # -------------------------------------------------------------
    # HOME LANDING HUB (EYE-CATCHING MODERN DASHBOARD)
    # -------------------------------------------------------------
    def show_home_hub(self):
        self.clear_container()
        self.btn_nav_home.configure(fg_color="#0284C7")
        self.btn_nav_mod1.configure(fg_color="#1E293B")
        self.btn_nav_mod2.configure(fg_color="#1E293B")

        hub_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        hub_frame.pack(fill="both", expand=True, padx=25, pady=20)

        # -------------------------------------------------------------
        # 1. TOP HERO BANNER WITH GLOW
        # -------------------------------------------------------------
        hero_card = ctk.CTkFrame(hub_frame, fg_color="#0F172A", corner_radius=16, border_width=1, border_color="#1E293B")
        hero_card.pack(fill="x", pady=(0, 16))

        hero_top = ctk.CTkFrame(hero_card, fg_color="transparent")
        hero_top.pack(fill="x", padx=25, pady=(18, 6))

        ctk.CTkLabel(
            hero_top, text="🚀 Welcome to SignVerse AI Platform", 
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"), 
            text_color="#F8FAFC"
        ).pack(side="left")

        # Pill Status Badge
        badge = ctk.CTkFrame(hero_top, fg_color="#064E3B", corner_radius=20)
        badge.pack(side="right")
        ctk.CTkLabel(badge, text=" ● SYSTEM ONLINE ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#34D399").pack(padx=10, pady=4)

        ctk.CTkLabel(
            hero_card, 
            text="Automated Deep Learning Framework for Bidirectional Sign Translation & Interactive Sign Language Education", 
            font=ctk.CTkFont(size=13), text_color="#94A3B8"
        ).pack(anchor="w", padx=25, pady=(0, 18))

        # -------------------------------------------------------------
        # 2. FOUR TOP STAT METRIC CARDS (EYE-CATCHY HIGHLIGHTS)
        # -------------------------------------------------------------
        stats_frame = ctk.CTkFrame(hub_frame, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 16))
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        stat_cards_data = [
            ("⚡ 60+ FPS", "Real-Time CPU Speed", "#0284C7"),
            ("🎯 97.62%", "Validation Accuracy", "#10B981"),
            ("📚 300 Glosses", "WLASL Daily Lexicon", "#8B5CF6"),
            ("🔒 100% Offline", "Zero Cloud Latency", "#F59E0B")
        ]

        for i, (val, title, color) in enumerate(stat_cards_data):
            c = ctk.CTkFrame(stats_frame, fg_color="#0F172A", corner_radius=12, border_width=1, border_color="#1E293B")
            c.grid(row=0, column=i, padx=5 if i not in (0, 3) else (0, 5) if i == 0 else (5, 0), sticky="ew")
            ctk.CTkLabel(c, text=val, font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack(anchor="w", padx=16, pady=(10, 2))
            ctk.CTkLabel(c, text=title, font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(anchor="w", padx=16, pady=(0, 10))

        # -------------------------------------------------------------
        # 3. INTERACTIVE MODULE SELECTION CARDS
        # -------------------------------------------------------------
        cards_grid = ctk.CTkFrame(hub_frame, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True)
        cards_grid.grid_columnconfigure((0, 1), weight=1)
        cards_grid.grid_rowconfigure(0, weight=1)

        # ------------------ CARD 1: MODULE 1 ------------------
        card1 = ctk.CTkFrame(cards_grid, fg_color="#0F172A", corner_radius=18, border_width=1, border_color="#0284C7")
        card1.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")

        # Header tag
        tag1 = ctk.CTkFrame(card1, fg_color="#0C4A6E", corner_radius=8)
        tag1.pack(anchor="w", padx=22, pady=(20, 4))
        ctk.CTkLabel(tag1, text="  💬 MODULE 1  ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8").pack(padx=4, pady=3)

        ctk.CTkLabel(card1, text="Communication Engine", font=ctk.CTkFont(size=20, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=22, pady=(2, 6))
        ctk.CTkLabel(card1, text="Enables bidirectional real-time communication between Deaf signers and hearing non-signers.", font=ctk.CTkFont(size=12), text_color="#94A3B8", wraplength=330, justify="left").pack(anchor="w", padx=22, pady=(0, 12))

        features1 = [
            ("🎥 Option 1", "Live Camera Sign-to-Text & SAPI5 Voice"),
            ("🤖 Option 2", "Text-to-3D Anatomical Avatar Mesh Translator"),
            ("🔄 FIFO Buffer", "30-Frame Rolling Window (1s Trajectory)"),
            ("🎯 Dynamic Filter", "0.55 (2-Hand) / 0.65 (1-Hand) Threshold")
        ]
        for f_tag, f_desc in features1:
            row_f = ctk.CTkFrame(card1, fg_color="transparent")
            row_f.pack(fill="x", padx=22, pady=3)
            ctk.CTkLabel(row_f, text=f"{f_tag}:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(side="left")
            ctk.CTkLabel(row_f, text=f" {f_desc}", font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(side="left")

        ctk.CTkButton(
            card1, text="Launch Communication Engine  ➔", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", 
            height=44, corner_radius=10,
            command=self.show_module1
        ).pack(fill="x", padx=22, pady=(22, 18), side="bottom")

        # ------------------ CARD 2: MODULE 2 ------------------
        card2 = ctk.CTkFrame(cards_grid, fg_color="#0F172A", corner_radius=18, border_width=1, border_color="#10B981")
        card2.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")

        # Header tag
        tag2 = ctk.CTkFrame(card2, fg_color="#064E3B", corner_radius=8)
        tag2.pack(anchor="w", padx=22, pady=(20, 4))
        ctk.CTkLabel(tag2, text="  🎓 MODULE 2  ", font=ctk.CTkFont(size=11, weight="bold"), text_color="#34D399").pack(padx=4, pady=3)

        ctk.CTkLabel(card2, text="LearnSign Learning Engine", font=ctk.CTkFont(size=20, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=22, pady=(2, 6))
        ctk.CTkLabel(card2, text="Interactive sign language learning platform with authentic human video demos & real-time AI scoring.", font=ctk.CTkFont(size=12), text_color="#94A3B8", wraplength=330, justify="left").pack(anchor="w", padx=22, pady=(0, 12))

        features2 = [
            ("🎬 Real Videos", "Authentic Human Signer Demo + 3D Mesh Toggle"),
            ("🔍 Instant Search", "Search & Learn any of the 300 WLASL Signs"),
            ("📐 Cosine Scorer", "Real-Time Joint Landmark Alignment (% Match)"),
            ("🎮 Dual Quizzes", "'Guess the Sign' Video Trivia & Live AI Challenge")
        ]
        for f_tag, f_desc in features2:
            row_f = ctk.CTkFrame(card2, fg_color="transparent")
            row_f.pack(fill="x", padx=22, pady=3)
            ctk.CTkLabel(row_f, text=f"{f_tag}:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#34D399").pack(side="left")
            ctk.CTkLabel(row_f, text=f" {f_desc}", font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(side="left")

        ctk.CTkButton(
            card2, text="Launch LearnSign Engine  ➔", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981", hover_color="#059669", 
            height=44, corner_radius=10,
            command=self.show_module2
        ).pack(fill="x", padx=22, pady=(22, 18), side="bottom")

    # -------------------------------------------------------------
    # LAUNCH MODULE 1
    # -------------------------------------------------------------
    def show_module1(self):
        self.clear_container()
        self.btn_nav_home.configure(fg_color="#1E293B")
        self.btn_nav_mod1.configure(fg_color="#0284C7")
        self.btn_nav_mod2.configure(fg_color="#1E293B")

        self.current_view = Module1View(self.main_container, self.model, self.label_map)
        self.current_view.pack(fill="both", expand=True)

    # -------------------------------------------------------------
    # LAUNCH MODULE 2 (PASSES MODEL & LABEL MAP)
    # -------------------------------------------------------------
    def show_module2(self):
        self.clear_container()
        self.btn_nav_home.configure(fg_color="#1E293B")
        self.btn_nav_mod1.configure(fg_color="#1E293B")
        self.btn_nav_mod2.configure(fg_color="#10B981")

        self.current_view = Module2View(self.main_container, self.model, self.label_map)
        self.current_view.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = SignVerseMasterApp()
    app.mainloop()
