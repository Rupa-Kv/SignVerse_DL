import os
import json
import customtkinter as ctk
import tensorflow as tf

from module1_engine import Module1View
from module2_engine import Module2View

# System Theme Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignVerseMasterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SignVerse AI Platform - Master Desktop Suite")
        self.geometry("1200x750")
        self.minsize(1050, 680)

        # Shared Model & Label Map Loading
        self.model = None
        self.label_map = {}
        self.load_model()

        # Active Module Views
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
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#0F172A")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Branding
        ctk.CTkLabel(self.sidebar, text="SignVerse AI", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00E5FF").grid(row=0, column=0, padx=20, pady=(25, 2), sticky="w")
        ctk.CTkLabel(self.sidebar, text="Unified Master Suite", font=ctk.CTkFont(size=12), text_color="#94A3B8").grid(row=1, column=0, padx=20, pady=(0, 25), sticky="w")

        # Nav Buttons
        self.btn_nav_home = ctk.CTkButton(
            self.sidebar, text="🏠 Home Dashboard", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", height=42, corner_radius=8, anchor="w",
            command=self.show_home_hub
        )
        self.btn_nav_home.grid(row=2, column=0, padx=15, pady=6, sticky="ew")

        self.btn_nav_mod1 = ctk.CTkButton(
            self.sidebar, text="💬 Module 1: Communication", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=42, corner_radius=8, anchor="w",
            command=self.show_module1
        )
        self.btn_nav_mod1.grid(row=3, column=0, padx=15, pady=6, sticky="ew")

        self.btn_nav_mod2 = ctk.CTkButton(
            self.sidebar, text="📚 Module 2: LearnSign Engine", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", height=42, corner_radius=8, anchor="w",
            command=self.show_module2
        )
        self.btn_nav_mod2.grid(row=4, column=0, padx=15, pady=6, sticky="ew")

        # Sidebar Footer Specs
        info_card = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=10)
        info_card.grid(row=7, column=0, padx=15, pady=20, sticky="ew")
        ctk.CTkLabel(info_card, text="Platform Specifications", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(info_card, text="• Benchmark: WLASL-300\n• Architecture: Conv1D+BiLSTM\n• Avatar: 3D Anatomical Mesh\n• AI Pose Evaluator: Cosine Sim", font=ctk.CTkFont(size=11), text_color="#94A3B8", justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        # -------------------------------------------------------------
        # MAIN CONTENT DISPLAY CONTAINER
        # -------------------------------------------------------------
        self.main_container = ctk.CTkFrame(self, fg_color="#020617", corner_radius=0)
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
            self.current_view = None

        for w in list(self.main_container.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass

    # -------------------------------------------------------------
    # HOME LANDING HUB
    # -------------------------------------------------------------
    def show_home_hub(self):
        self.clear_container()
        self.btn_nav_home.configure(fg_color="#0284C7")
        self.btn_nav_mod1.configure(fg_color="#1E293B")
        self.btn_nav_mod2.configure(fg_color="#1E293B")

        hub_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        hub_frame.pack(fill="both", expand=True, padx=30, pady=30)

        # Welcome Banner
        banner = ctk.CTkFrame(hub_frame, fg_color="#0F172A", corner_radius=15, border_width=1, border_color="#1E293B")
        banner.pack(fill="x", pady=(0, 25))

        ctk.CTkLabel(banner, text="Welcome to SignVerse AI Platform", font=ctk.CTkFont(size=22, weight="bold"), text_color="#00E5FF").pack(anchor="w", padx=25, pady=(20, 4))
        ctk.CTkLabel(banner, text="Automated Deep Learning Framework for Bidirectional Sign Language Translation & Interactive Learning", font=ctk.CTkFont(size=13), text_color="#94A3B8").pack(anchor="w", padx=25, pady=(0, 20))

        # Module Choice Cards Grid (2 Big Columns)
        cards_grid = ctk.CTkFrame(hub_frame, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True)
        cards_grid.grid_columnconfigure((0, 1), weight=1)
        cards_grid.grid_rowconfigure(0, weight=1)

        # CARD 1: MODULE 1
        card1 = ctk.CTkFrame(cards_grid, fg_color="#0F172A", corner_radius=18, border_width=1, border_color="#0284C7")
        card1.grid(row=0, column=0, padx=(0, 15), pady=0, sticky="nsew")

        ctk.CTkLabel(card1, text="MODULE 1", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=22, pady=(22, 2))
        ctk.CTkLabel(card1, text="Communication Engine", font=ctk.CTkFont(size=20, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=22, pady=(0, 10))
        
        ctk.CTkLabel(card1, text="Enables real-time two-way communication between Deaf signers and hearing non-signers.", font=ctk.CTkFont(size=12), text_color="#94A3B8", wraplength=340, justify="left").pack(anchor="w", padx=22, pady=(0, 15))

        features1 = [
            "• Option 1: Live Sign-to-Text & Speech Synthesis",
            "• Option 2: Text-to-3D Anatomical Avatar Mesh",
            "• 60+ FPS Real-Time Thread-Safe Camera Stream",
            "• Dynamic Thresholding (0.55-0.65) for Subtle Signs"
        ]
        for feat in features1:
            ctk.CTkLabel(card1, text=feat, font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(anchor="w", padx=22, pady=3)

        ctk.CTkButton(
            card1, text="Launch Communication Engine ➔", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0284C7", hover_color="#0369A1", height=45, corner_radius=10,
            command=self.show_module1
        ).pack(fill="x", padx=22, pady=(25, 22))

        # CARD 2: MODULE 2
        card2 = ctk.CTkFrame(cards_grid, fg_color="#0F172A", corner_radius=18, border_width=1, border_color="#10B981")
        card2.grid(row=0, column=1, padx=(15, 0), pady=0, sticky="nsew")

        ctk.CTkLabel(card2, text="MODULE 2", font=ctk.CTkFont(size=12, weight="bold"), text_color="#34D399").pack(anchor="w", padx=22, pady=(22, 2))
        ctk.CTkLabel(card2, text="LearnSign Learning Engine", font=ctk.CTkFont(size=20, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=22, pady=(0, 10))

        ctk.CTkLabel(card2, text="Interactive sign language learning platform with live AI pose evaluation and score feedback.", font=ctk.CTkFont(size=12), text_color="#94A3B8", wraplength=340, justify="left").pack(anchor="w", padx=22, pady=(0, 15))

        features2 = [
            "• Guided Lesson Curriculum (Categorized Signs)",
            "• Live AI Practice with Cosine Pose Evaluator",
            "• Instant Visual Feedback Badges (Perfect/Good)",
            "• Interactive Skill Quiz with Streaks & Points"
        ]
        for feat in features2:
            ctk.CTkLabel(card2, text=feat, font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(anchor="w", padx=22, pady=3)

        ctk.CTkButton(
            card2, text="Launch LearnSign Engine ➔", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981", hover_color="#059669", height=45, corner_radius=10,
            command=self.show_module2
        ).pack(fill="x", padx=22, pady=(25, 22))

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
    # LAUNCH MODULE 2
    # -------------------------------------------------------------
    def show_module2(self):
        self.clear_container()
        self.btn_nav_home.configure(fg_color="#1E293B")
        self.btn_nav_mod1.configure(fg_color="#1E293B")
        self.btn_nav_mod2.configure(fg_color="#10B981")

        self.current_view = Module2View(self.main_container, self.label_map)
        self.current_view.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = SignVerseMasterApp()
    app.mainloop()
