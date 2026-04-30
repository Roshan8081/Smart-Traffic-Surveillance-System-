<<<<<<< HEAD
# 🚦 Smart Traffic Violation Detection System

AI-powered full-stack system for detecting traffic violations using computer vision and OCR.

---

## 🔥 Features

* 🚗 Vehicle detection using YOLOv8
* 🔁 Multi-object tracking with persistent IDs
* ⚡ Speed estimation & overspeed detection
* 🚦 Red Light Violation Detection (RLVD)
* 🔍 Automatic Number Plate Recognition (ANPR)
* 🖼️ Violation evidence capture (images + plates)
* 🌐 Backend API (Node.js + MongoDB)
* 📊 Frontend dashboard (React + Vite)
* ⚡ GPU acceleration support (CUDA)

---

## 🧠 Architecture

```
Video → YOLO → Tracking → Speed + RLVD + ANPR → Violation Manager
                                                  ↓
                                            Save Images
                                                  ↓
                                          Backend API
                                                  ↓
                                             Dashboard
```

---

## 📁 Project Structure

```
smart_traffic_system/
├── detection/
├── backend/
├── frontend/
├── assets/
├── run_project.ps1
├── requirements.txt
└── README.md
```

---

## 🎥 Dataset (Traffic Videos)

Due to GitHub size limits, videos are not included.

👉 Download sample videos from:
**[https://drive.google.com/drive/folders/1v1PQ_gltk61enif7Q533D1FKFlLU2bTV?usp=sharing]**

After downloading, place them in:

```
detection/data/
```

---

## 🧠 YOLO Model (IMPORTANT)

Download YOLOv8 model manually:

👉 https://github.com/ultralytics/assets/releases

Place it in project root or detection folder:

```
yolov8n.pt
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd smart_traffic_system

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Run the Project

### Option 1 (Recommended - Windows)

```powershell
.\run_project.ps1
```

---

### Option 2 (Manual)

```bash
# Run detection
python detection/main.py

# Run backend
cd backend
npm install
npm run dev

# Run frontend
cd frontend
npm install
npm run dev
```

---

## ⚠️ PowerShell Fix (if script fails)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 📊 Results

### Detection & Tracking

![Detection](assets/detection.png)

### Speed Detection

![Speed](assets/speed.png)

### Red Light Violation

![RLVD](assets/rlvd.png)

### ANPR

![ANPR](assets/anpr.png)

---

## 🧪 Configuration

* Speed limit → `SPEED_LIMIT_KMH`
* Detection interval → YOLO config
* OCR frequency → ANPR config

---

## 🌐 Backend API

POST:

```
/api/violations
```

GET:

```
/api/violations
```

---

## 🛠 Tech Stack

* Python (OpenCV, YOLOv8, EasyOCR)
* Node.js + Express + MongoDB
* React + Vite

---

## 🚀 Future Scope

* Real traffic signal integration
* Advanced ANPR accuracy
* Multi-camera support
* Cloud deployment

---

## 📜 License

MIT License
=======
# Smart-Traffic-Surveillance-System-
Building an AI-powered system to automate real-time traffic monitoring, including vehicle detection, number plate recognition, and speed tracking;achieving 92% vehicle detection accuracy and 85% plate recognition precision. Integrating YOLOv5 object detection with OpenCV for live video feeds.
>>>>>>> 09c9b8b9f31b61c333999bb0ca91049acb6113a5
