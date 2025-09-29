# 🦺 SH17 Safety Equipment Detection (YOLOv11 + Streamlit)

This repository contains my **YOLOv11 object detection model** trained on the 
[SH17 PPE dataset](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection) 
(created by *mugheesahmad* on Kaggle).  

👉 **Note:** The dataset belongs to the original author.  
👉 The **model training, fine-tuning, and deployment app** are my own work.

---

## ✨ Features
- Detects **12 PPE classes**: helmet, gloves, glasses, vest, shoes, person, etc.
- Supports:
  - 📷 **Image Inference**
  - 🎬 **Video Inference**
  - 📸 **Webcam Snapshot**
  - 🔴 **Live Webcam Detection**
- Adjustable:
  - Confidence threshold
  - Inference size (640–1280 px)
  - Class filters
- Option to **save annotated images/videos** with bounding boxes
- FPS overlay for video & live webcam

---

## ⚙️ Tech Stack
- **Model:** YOLOv11 (Ultralytics)
- **Dataset:** SH17 PPE dataset (Kaggle, Ahmad Mughees)
- **Framework:** Streamlit + OpenCV + Ultralytics
- **Deployment:** 
  - [Streamlit Community Cloud](https://share.streamlit.io) (CPU only)  
  - [Hugging Face Spaces](https://huggingface.co/spaces) (GPU available)

---

## 🚀 Demo
- 🔗 **Streamlit App**: [coming soon]  
- 🔗 **Hugging Face Space**: [coming soon]  

---

## License
- Code and trained YOLOv11 model weights in this repository are released under the **MIT License**.  
- The dataset used for training is the **[SH17 PPE dataset](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)** by *mugheesahmad*, which is licensed under **CC BY-NC-SA 4.0**.  
- This repository does **not** include the dataset itself. Please download it directly from Kaggle.  

