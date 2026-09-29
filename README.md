# 🌿 PlantGuard AI — Leaf Disease Detection

## Project Overview

An AI-powered web application that detects plant leaf diseases using deep learning. The model is **trained once** and saved, then loaded at server startup for instant predictions — no repeated training needed.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   train_model.py │───►│  model/           │    │                     │
│   (run ONCE)     │    │  ├ model.keras    │◄───│   app.py (Flask)    │
│                  │    │  └ class_labels   │    │   Loads model once  │
└─────────────────┘    └──────────────────┘    │   at startup        │
                                                └────────┬────────────┘
                                                         │ REST API
                                                ┌────────▼────────────┐
                                                │  Frontend (HTML/    │
                                                │  CSS / JavaScript)  │
                                                │  Upload → Predict   │
                                                └─────────────────────┘
```

## How to Use

### Step 1: Train the Model (One Time Only)
```bash
python train_model.py
```

### Step 2: Run the Web App
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

*(Optional)* To enable live conversational reasoning in the AI Chatbot, set your Gemini API key:
```bash
set GEMINI_API_KEY=your_key_here     # Windows Command Prompt
$env:GEMINI_API_KEY="your_key_here"  # Windows PowerShell
python app.py
```
*(Note: If no API key is provided, the AI Chatbot automatically runs in intelligent offline pathology mode.)*

### Step 3: Upload & Diagnose
Upload a leaf photo, take a picture with your webcam/camera, or pick a sample leaf for an instant deep learning diagnosis. You can also consult the **AI Plant Pathologist** for organic/chemical remedies or check the **Live Weather & Disease Risk Tracker**!

## Key Features

- 🌿 **Deep Learning Diagnosis**: 15 foliar conditions across Tomato, Potato & Pepper crops.
- 💬 **AI Plant Pathologist Chatbot**: Dual-mode chatbot (embedded full-section and floating companion) for crop consultations, custom treatment schedules, and bio-security protocols.
- 🌤️ **Live Weather & Disease Risk Tracker**: Open-Meteo integration predicting fungal spore risk based on local temperature, humidity, wind, and rain forecasts.
- 📋 **Clinical Dossiers & PDF Certificate Export**: Comprehensive encyclopedia and one-click printable diagnostic certificates.

## Supported Diseases (15 Classes)

| Plant | Conditions |
|-------|-----------|
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫑 Pepper Bell | Bacterial Spot, Healthy |

## Tech Stack

- **Model**: TensorFlow/Keras with MobileNetV2 transfer learning
- **Backend**: Flask (Python) with REST API & Gemini 2.0 AI integration
- **Frontend**: HTML5, Modern Vanilla CSS, Dynamic Vanilla JavaScript
- **Dataset**: PlantVillage (~20,000+ images)
