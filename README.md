# 🤟 SignVerse - Deep Learning Accessibility Platform for Deaf & Mute Empowerment

**Project Location**: `C:\Users\Rupa\Documents\SignVerse`

SignVerse is an AI-powered accessibility system designed to facilitate bidirectional communication, interactive sign language learning, and multi-sensory cinema entertainment for Deaf, Hard of Hearing (DHH), and speech-impaired individuals.

---

## 📁 Module 1: Communication Engine (`SignComm`)

Module 1 consists of two core deep learning components:

### 1. Sign-to-Text (+ Text-to-Speech Read Aloud)
- **Landmark Extraction**: Uses **MediaPipe Hands & Pose** to extract 21 3D hand keypoints per hand `(x, y, z)`.
- **Deep Learning Model**: Sequential LSTM / Dense Neural Network trained on extracted landmark frame sequences.
- **Webcam Inference**: Real-time sign prediction with live text output and Pyttsx3 / Speech Synthesis narration.

### 2. Text-to-Sign Engine
- Converts typed or spoken text into standardized sign tokens and gesture sequence representations.
- Maps words and alphabets to sign cards / landmark representations.

---

## 📊 Recommended Datasets for Sign Language Recognition

1. **WLASL (Word-Level American Sign Language)** *(Primary Recommendation)*:
   - **Download Link**: [GitHub - WLASL Dataset](https://github.com/dxli94/WLASL)
   - **Size**: 2,000 words across 21,000 video clips.
   - **Use Case**: Excellent for training word-level sign recognition models.

2. **ASL Citizen**:
   - **Size**: 83,000 video clips covering 2,731 ASL signs performed by 52 signers.
   - **Use Case**: Modern dataset with high signer diversity and clean backgrounds.

3. **MS-ASL (Microsoft ASL Dataset)**:
   - **Size**: 25,510 video clips covering 1,000 signs.

4. **INCLUDE Dataset (For Indian Sign Language)**:
   - **Size**: 263 signs across 4,287 videos recorded from deaf school students.

---

## 🛠️ Installation & Setup

```bash
# Navigate to project directory
cd C:\Users\Rupa\Documents\SignVerse

# Install dependencies
pip install -r requirements.txt

# Run Module 1 Interactive Suite
python main_module1.py
```
