import os
import json
import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="SignVerse - AI Communication Platform",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean Professional Dark Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 5px;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .nav-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# MediaPipe 3D Landmark Extractor Class
class StreamlitMediaPipeExtractor:
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
class Streamlit3DAvatarEngine:
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
        canvas = np.zeros((500, 700, 3), dtype=np.uint8)
        canvas.fill(20)
        h, w = canvas.shape[:2]
        
        cv2.rectangle(canvas, (0, 0), (w, 65), (35, 35, 35), -1)
        cv2.putText(canvas, f"SignVerse 3D Avatar: {gloss_name.upper()}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        cv2.putText(canvas, f"Frame {frame_num}/{total_frames} | Skeleton Joint Motion", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        head_center = (w // 2, 120)
        neck = (w // 2, 160)
        left_shoulder = (w // 2 - 80, 185)
        right_shoulder = (w // 2 + 80, 185)
        spine_base = (w // 2, 280)
        
        cv2.circle(canvas, head_center, 32, (0, 200, 255), 3)
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
            l_wrist_px = (int(left_shoulder[0] + left_pts[0,0]*130), int(left_shoulder[1] + 65 + left_pts[0,1]*130))
            cv2.line(canvas, left_shoulder, l_wrist_px, (100, 255, 100), 3)
            l_px_list = []
            for pt in left_pts:
                px = (int(l_wrist_px[0] + pt[0]*100), int(l_wrist_px[1] + pt[1]*100))
                l_px_list.append(px)
                cv2.circle(canvas, px, 3, (0, 255, 0), -1)
            for p1, p2 in connections:
                cv2.line(canvas, l_px_list[p1], l_px_list[p2], (50, 200, 50), 2)
                
        right_pts = pts_vector[63:126].reshape((21, 3))
        if np.any(right_pts):
            r_wrist_px = (int(right_shoulder[0] + right_pts[0,0]*130), int(right_shoulder[1] + 65 + right_pts[0,1]*130))
            cv2.line(canvas, right_shoulder, r_wrist_px, (255, 150, 100), 3)
            r_px_list = []
            for pt in right_pts:
                px = (int(r_wrist_px[0] + pt[0]*100), int(r_wrist_px[1] + pt[1]*100))
                r_px_list.append(px)
                cv2.circle(canvas, px, 3, (255, 100, 0), -1)
            for p1, p2 in connections:
                cv2.line(canvas, r_px_list[p1], r_px_list[p2], (220, 100, 0), 2)
                
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Voice Text-To-Speech Function
def speak_audio(text):
    if TTS_AVAILABLE:
        try:
            engine = pyttsx3.init('sapi5')
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

# Cache Model Weights
@st.cache_resource
def load_signverse_model():
    model_path = "saved_models/best_signverse_model_300.keras"
    label_path = "preprocessed_data_300/label_map_300.json"
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        model = tf.keras.models.load_model(model_path)
        with open(label_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}
        return model, label_map
    return None, None

# Main Web App Logic
def main():
    # Session State Page Controller
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
        
    # Header Title
    st.markdown('<div class="main-header">SignVerse AI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bidirectional Sign Language Communication & 3D Avatar System</div>', unsafe_allow_html=True)

    # Sidebar Navigation Buttons
    st.sidebar.title("Navigation Hub")
    if st.sidebar.button("Home Dashboard", use_container_width=True):
        st.session_state.current_page = "home"
    if st.sidebar.button("Module 1: Communication Engine", use_container_width=True):
        st.session_state.current_page = "module1_select"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Overview:**")
    st.sidebar.markdown("- Benchmark: WLASL-300")
    st.sidebar.markdown("- Model: Conv1D + BiLSTM")
    st.sidebar.markdown("- Feature Vector: 126 Landmarks")

    # PAGE 1: HOME DASHBOARD
    if st.session_state.current_page == "home":
        st.subheader("Welcome to SignVerse")
        st.write("Select a module below to launch bidirectional sign language recognition and 3D Avatar translation.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="nav-card">
                <h3>Module 1: SignVerse Communication Engine</h3>
                <p>Bidirectional Sign-to-Text/Audio recognition and Text-to-3D Avatar translation for 300 sign words.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Launch Module 1", type="primary", use_container_width=True):
                st.session_state.current_page = "module1_select"
                st.rerun()
                
        with col2:
            st.markdown("""
            <div class="nav-card">
                <h3>Module 2: Advanced NLP Context Engine</h3>
                <p>Grammar refinement and smart contextual sentence prediction (Coming Soon).</p>
            </div>
            """, unsafe_allow_html=True)
            st.button("Module 2 (Locked)", disabled=True, use_container_width=True)

    # PAGE 2: MODULE 1 SELECTION HUB
    elif st.session_state.current_page in ["module1_select", "sign_to_text", "text_to_sign"]:
        st.subheader("Module 1: SignVerse Communication Engine")
        st.write("Choose a feature below to begin demonstration:")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Option 1: Sign-to-Text / Audio (Live Camera)", type="primary" if st.session_state.current_page == "sign_to_text" else "secondary", use_container_width=True):
                st.session_state.current_page = "sign_to_text"
                st.rerun()
                
        with col_btn2:
            if st.button("Option 2: Text-to-Sign (3D Avatar Translator)", type="primary" if st.session_state.current_page == "text_to_sign" else "secondary", use_container_width=True):
                st.session_state.current_page = "text_to_sign"
                st.rerun()
                
        st.markdown("---")

        # FEATURE 1: SIGN TO TEXT
        if st.session_state.current_page == "sign_to_text":
            st.markdown("### Feature 1: Live Sign Language Recognition & Speech Output")
            st.write("Perform sign language gestures in front of your camera. The Conv1D+BiLSTM neural network predicts words and forms full sentences with voice speech output.")
            
            model, label_map = load_signverse_model()
            if model is None:
                st.error("Trained model weights not found in `saved_models/best_signverse_model_300.keras`.")
                return

            c1, c2 = st.columns([2, 1])
            with c2:
                st.markdown("#### Live Metrics")
                pred_slot = st.empty()
                conf_slot = st.empty()
                sent_slot = st.empty()
                
                st.markdown("#### Controls")
                if st.button("Speak Active Sentence", use_container_width=True):
                    if "current_words" in st.session_state and st.session_state.current_words:
                        speak_audio(" ".join(st.session_state.current_words))
                if st.button("Clear Active Sentence", use_container_width=True):
                    st.session_state.current_words = []
                    
            with c1:
                run_cam = st.checkbox("Open Live Camera Stream", value=False)
                cam_slot = st.empty()
                
                if run_cam:
                    cap = cv2.VideoCapture(0)
                    extractor = StreamlitMediaPipeExtractor()
                    sequence_buffer = []
                    if "current_words" not in st.session_state:
                        st.session_state.current_words = []
                    completed_sentences = []
                    last_pred = ""
                    pred_conf = 0.0
                    last_hand_time = time.time()
                    
                    while run_cam and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Camera disconnected.")
                            break
                            
                        frame = cv2.flip(frame, 1)
                        feats, annotated_frame, detected_count = extractor.extract_landmarks(frame)
                        
                        if detected_count > 0:
                            last_hand_time = time.time()
                        else:
                            if st.session_state.current_words and (time.time() - last_hand_time > 2.0):
                                sentence_str = " ".join(st.session_state.current_words) + "."
                                completed_sentences.append(sentence_str)
                                speak_audio(sentence_str)
                                st.session_state.current_words = []
                                last_pred = ""

                        sequence_buffer.append(feats)
                        if len(sequence_buffer) > 30:
                            sequence_buffer.pop(0)

                        if len(sequence_buffer) == 30 and detected_count > 0:
                            seq_input = np.array([sequence_buffer], dtype=np.float32)
                            probs = model.predict(seq_input, verbose=0)[0]
                            pred_idx = int(np.argmax(probs))
                            pred_conf = float(probs[pred_idx])
                            pred_word = label_map.get(pred_idx, "Unknown")
                            
                            if pred_conf > 0.65 and pred_word != last_pred:
                                last_pred = pred_word
                                if not st.session_state.current_words or st.session_state.current_words[-1] != pred_word:
                                    st.session_state.current_words.append(pred_word)

                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        cam_slot.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        pred_slot.markdown(f"**Live Sign**: `{last_pred.upper() if last_pred else 'Waiting...'}`")
                        conf_slot.markdown(f"**Confidence**: `{pred_conf*100:.1f}%`")
                        sent_slot.markdown(f"**Active Sentence**: `{ ' '.join(st.session_state.current_words) }`")
                        
                    cap.release()
                    extractor.hands.close()

        # FEATURE 2: TEXT TO SIGN
        elif st.session_state.current_page == "text_to_sign":
            st.markdown("### Feature 2: Text-to-Sign 3D Skeleton Avatar Translator")
            st.write("Type an English sentence below to render full-body and dual-hand 3D skeleton gesture animations.")
            
            user_text = st.text_input("Enter text sentence:", "hello doctor please help computer")
            fps = st.slider("Avatar FPS Speed:", min_value=10, max_value=60, value=25)
            
            if st.button("Render 3D Avatar Gestures", type="primary"):
                engine = Streamlit3DAvatarEngine()
                words = [w.strip().upper() for w in user_text.split() if w.strip()]
                
                if not words:
                    st.warning("Please enter text.")
                    return
                    
                st.info(f"Rendering {len(words)} sign words: {', '.join(words)}")
                avatar_slot = st.empty()
                extractor = StreamlitMediaPipeExtractor()
                
                for word in words:
                    v_path = engine.gloss_map.get(word)
                    if not v_path:
                        st.error(f"Word '{word}' is not in the WLASL dataset index.")
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
                            img = engine.draw_skeleton(pts, word, idx+1, total_f)
                            avatar_slot.image(img, channels="RGB", use_container_width=True)
                            time.sleep(1.0 / fps)
                        time.sleep(0.3)
                        
                extractor.hands.close()
                st.success("3D Avatar Gesture Rendering Finished!")

if __name__ == "__main__":
    main()
