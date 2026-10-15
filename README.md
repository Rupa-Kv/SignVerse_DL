# ✨ SignVerse AI — Sign Language Recognition & Learning Platform

**An AI-powered desktop application for sign language recognition, communication assistance, and interactive sign language learning.**

SignVerse AI integrates deep learning, computer vision, and an interactive graphical user interface to support sign-to-text recognition, speech output, sign demonstrations, and learner assessment.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Deep Learning Model](#-deep-learning-model)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset and Model Setup](#-dataset-and-model-setup)
- [Running the Application](#-running-the-application)
- [Application Modules](#-application-modules)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

SignVerse AI is a Python-based desktop platform designed to make sign language recognition and learning more accessible through artificial intelligence.

The application combines a deep-learning recognition pipeline with an interactive learning environment. It processes hand landmarks extracted from live camera frames and uses a trained neural network to predict signs from temporal sequences.

The platform consists of two primary modules:

- **Communication Engine:** Supports live sign-to-text recognition, speech output, and text-to-sign visualization.
- **LearnSign Engine:** Provides sign demonstrations, guided lessons, live practice, and interactive quizzes with learner progress tracking.

The project uses the **WLASL-300 vocabulary subset**, which contains 300 American Sign Language (ASL) gloss classes.

## 🎯 Key Features

### 💬 Module 1: Communication Engine

- Real-time webcam input for sign recognition.
- Hand landmark extraction using MediaPipe.
- Temporal sequence-based sign classification.
- Predicted sign labels and confidence scores.
- Sentence buffering and text-to-speech output.
- Text input for sign visualization.
- Skeleton-based hand visualization for the avatar interface.

### 🎓 Module 2: LearnSign Engine

- Interactive sign language learning interface.
- Search functionality for vocabulary signs.
- Human signer video demonstrations from the WLASL dataset.
- Skeleton visualization mode.
- Live webcam-based practice interface.
- Interactive vocabulary quizzes.
- Learner profiles and progress tracking.
- Quiz scores, mastered signs, and mistake history.

### 🖥️ User Interface

- Dark-themed desktop interface built with CustomTkinter.
- Central dashboard for navigating application modules.
- Separate interfaces for communication and learning.
- Reusable model and label-map loading through the main application.

---

## 🏗️ System Architecture

The application follows a modular architecture that separates model inference, user-interface logic, and learning functionality.

```text
                    SIGNVERSE AI
                          |
                SignVerse Master App
                          |
             Shared Model and Label Map
                          |
             +------------+------------+
             |                         |
      Communication Engine       LearnSign Engine
             |                         |
       Webcam Input              Learning Interface
             |                         |
       MediaPipe Hands          Sign Video Demos
             |                         |
       126-D Landmark           Live Practice
          Features                    |
             |                   Quiz Engine
       30-Frame Sequence              |
             |                  Learner Profiles
       Trained Keras Model
             |
       Sign Classification
             |
       Text and Speech Output
```

### Processing Pipeline

1. Capture video frames from the webcam.
2. Extract hand landmarks using MediaPipe.
3. Normalize and organize the landmarks into a feature vector.
4. Build a sequence of 30 consecutive frames.
5. Pass the sequence to the trained deep-learning model.
6. Obtain class probabilities and identify the predicted gloss.
7. Display the prediction and confidence score.
8. Optionally convert the recognized text into speech.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Desktop GUI | CustomTkinter |
| Computer Vision | OpenCV |
| Hand Landmark Detection | MediaPipe |
| Deep Learning | TensorFlow / Keras |
| Numerical Processing | NumPy |
| Image Processing | Pillow |
| Text-to-Speech | pyttsx3 |
| Dataset | WLASL-300 |
| Model Format | Keras `.keras` |
| Development Notebook | Jupyter Notebook |

---

## 🧠 Deep Learning Model

SignVerse AI uses a trained Keras model to classify temporal sequences of hand landmarks.

### Input Representation

| Parameter | Value |
|---|---|
| Sequence length | 30 frames |
| Landmarks per hand | 21 |
| Coordinates per landmark | 3 (x, y, z) |
| Maximum number of hands | 2 |
| Features per frame | 126 |
| Output classes | 300 |

The expected model input shape is:

```text
(batch_size, 30, 126)
```

The model outputs scores for 300 sign classes. The application selects the class with the highest predicted score and maps it to the corresponding vocabulary label.

The training notebook documents the model development and evaluation workflow. The saved model is intended to be used for inference by the desktop application.

**Note:** Recognition performance depends on the training data, camera conditions, hand visibility, signing variation, and consistency between training-time and inference-time preprocessing.

---

## 📂 Project Structure

Organize the repository as follows:

```text
SignVerse-AI/
│
├── SignVerse_Master_App.py
├── module1_engine.py
├── module2_engine.py
├── requirements.txt
├── README.md
│
├── SignVerse_Module1_WLASL300_DeepLearning.ipynb
│
├── saved_models/
│   └── best_signverse_model_300.keras
│
├── preprocessed_data_300/
│   └── label_map_300.json
│
└── DATASET_WLASL/
    ├── WLASL_v0.3.json
    └── videos/
        └── ...
```

**Important:** The directory names shown above must match the paths used by your application. The dataset videos, label map, and trained model must be available at the expected locations before running the corresponding features.

Do not upload large datasets or model files to GitHub without checking repository size limits and dataset redistribution permissions.

---

## ⚙️ Installation

### Prerequisites

- Python installed on your system.
- A working webcam for live recognition and practice.
- A compatible TensorFlow and MediaPipe environment.
- Git for cloning the repository.

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SignVerse-AI
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version supported by your selected TensorFlow and MediaPipe releases. If installation fails, resolve the package-version compatibility issues before running the application.

---

## 📦 Dataset and Model Setup

The project uses the WLASL dataset for ASL sign vocabulary.

Before launching the application, ensure the following resources are available:

1. The trained Keras model: `best_signverse_model_300.keras`
2. The class label mapping: `label_map_300.json`
3. The WLASL metadata file: `WLASL_v0.3.json`
4. The corresponding video files required for sign demonstrations.

Place each resource in the directory expected by the code.

If you are training the model from scratch, open the notebook:

```text
SignVerse_Module1_WLASL300_DeepLearning.ipynb
```

Run the notebook in an environment containing the required dependencies and dataset. Ensure that the preprocessing and label mapping used during training match those used during inference.

---

## ▶️ Running the Application

Once the dependencies, model, label map, and required dataset resources are configured, run:

```bash
python SignVerse_Master_App.py
```

The main dashboard provides access to the available modules.

### Using Module 1

1. Open the Communication Engine.
2. Select the live camera recognition option.
3. Allow webcam access through your operating system.
4. Perform a sign in front of the camera.
5. Observe the predicted gloss and confidence score.
6. Use the speech feature to hear the recognized text.
7. Use the text-to-sign interface to explore the avatar visualization.

### Using Module 2

1. Open the LearnSign Engine.
2. Browse the lesson curriculum or search for a sign.
3. View the available signer demonstration.
4. Explore the visualization mode.
5. Use the practice interface and quizzes.
6. Review the available learner progress and quiz history.

---

## ⚠️ Limitations

- The current recognition system is designed around a fixed vocabulary of 300 ASL gloss classes.
- It does not provide unrestricted sign language translation or complete grammatical interpretation.
- Accuracy can vary with lighting, camera placement, signing speed, hand occlusion, and signer differences.
- The avatar interface currently provides skeleton-style visualization rather than a fully articulated 3D anatomical model.
- The live practice similarity score requires a validated reference-sign representation before it can be treated as a meaningful measure of signing accuracy.
- Runtime performance depends on the hardware and software environment.
- The application should not be treated as a substitute for qualified sign language interpretation in critical communication settings.

---

## 🔮 Future Enhancements

- Improve recognition stability with temporal smoothing and prediction filtering.
- Validate the model using leakage-free train, validation, and test splits.
- Add reference-based temporal alignment for sign practice evaluation.
- Develop a fully articulated 3D avatar with realistic hand and arm motion.
- Expand support for larger vocabularies and additional sign languages.
- Improve sentence-level translation and grammatical modeling.
- Add more robust learner analytics and progress visualization.
- Improve error handling, resource cleanup, and deployment portability.
- Package the application for easier installation on supported operating systems.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test the application and relevant model workflows.
5. Submit a pull request describing your changes.

For substantial changes, include the problem being addressed, the implementation approach, and any relevant testing results.

---

## 📄 License

No license has been specified for this project yet.

Before distributing the source code, trained model, or dataset, add an appropriate license and verify the applicable licensing and redistribution terms of all third-party components and datasets.

---

## 👨‍💻 Project Information

**Project:** SignVerse AI  
**Domain:** Artificial Intelligence, Deep Learning, Computer Vision, and Assistive Technology  
**Primary Dataset:** WLASL-300  
**Application Type:** Python Desktop Application

*SignVerse AI aims to support accessible communication and interactive sign language learning through computer vision and deep learning.*
