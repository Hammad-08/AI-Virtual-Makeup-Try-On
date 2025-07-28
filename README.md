
# 💄 AI Virtual Makeup Try-On Application

A real-time desktop-based virtual makeup try-on system built with **Python**, **MediaPipe**, **OpenCV**, and **PySide6**. It overlays virtual cosmetics like lipstick, eyeliner, and foundation on a user’s face using a **live video feed from an external USB camera**.

---

## 📁 Project Files

- `main.py` – Application entry point
- `requirements.txt` – Required Python libraries
- `environment.yml` – Conda environment setup file
- `Dockerfile` – Docker configuration (optional)
- Additional files: `camera.py`, UI assets, helper scripts

---

## ⚙️ System Requirements

- **Python 3.11+**
- **External USB Camera** (⚠️ *Required – application is coded specifically for external webcam*)
- **C++ Build Tools** installed (for compiling dependencies like `mediapipe` and `opencv`)
- **Windows 10/11** or **macOS/Linux** with Python support

---

## 🚀 Installation Guide

### 🔹 Option 1: Python Virtual Environment

```bash
python -m venv venv
./venv/Scripts/activate     # On Windows
source ./venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
python ./main.py
```

### 🔹 Option 2: Conda Environment (Optional)

```bash
conda env create -f environment.yml
conda activate virtual-makeup
python ./main.py
```

### 🔹 Option 3: Docker (If You Prefer Containerized Setup)

```bash
docker build -t virtual-makeup-app .
docker run -it --rm --device=/dev/video0 virtual-makeup-app
```

> ⚠️ Docker users: You must connect an **external USB camera** and expose the correct device inside the container (e.g., `/dev/video0` on Linux).

---

## 🎥 External Camera Requirement

⚠️ **IMPORTANT:**  
This application is designed to work **only with an external USB webcam**. Built-in laptop cameras may not function correctly due to specific video stream handling in the code.

- Connect your external webcam before launching the application.
- Ensure it is recognized by the system (check via device manager or `lsusb`).
- If multiple cameras are attached, the default camera index must point to the external one (modify code if needed).

---

## ❗ Troubleshooting

- **C++ Build Tools Error**:  
  If you face installation errors with `mediapipe`, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

- **Camera Not Working**:  
  - Ensure your external camera is plugged in and working.  
  - Verify the correct camera index in `camera.py`.

---

## ✨ Features

- Real-time facial landmark detection using MediaPipe
- Makeup product overlays: lipstick, eyeliner, eyeshadow, foundation, eyebrow
- Interactive PySide6 GUI
- Product intensity adjustment and layering

---

## 👥 Authors

- **Abdul Ahad**  
- **Hammad Yasir**  
- **Ahmad Raza**  
**Supervisor**: Dr. Rehan Ashraf  
**Co-Supervisor**: Dr. Sajida Parveen  
**National Textile University, Faisalabad**  
**BS Artificial Intelligence – Batch 2021–2025**

---

## 📜 License

This project is for academic purposes only. All rights reserved by the authors and the NTU Department of Computer Science.
