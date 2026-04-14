# 🦅 Bird vs Drone Detection AI

An AI-based system for **classification and detection of birds and drones** using **CNN and YOLOv8**, deployed with a **Streamlit web application** for real-time interaction.

---

## 📌 Features

- 🧠 Image Classification using CNN (97.21% accuracy)
- 📦 Object Detection using YOLOv8
- 🌐 Interactive UI built with Streamlit
- ⚡ Real-time image upload and prediction
- 🎯 Detection with bounding boxes and confidence scores

---

## 🧠 Models Used

### 🔹 CNN (Custom + Transfer Learning)
- Used for image classification (Bird vs Drone)
- Achieved **97.21% accuracy**
- Fine-tuned using pretrained weights

### 🔹 YOLOv8
- Used for object detection
- Detects and localizes birds and drones in images
- Outputs bounding boxes with labels

---

## 📂 Project Structure
project3/
│
├── app.py # Streamlit application
├── model.py # CNN model
├── model2.py # Transfer learning model
├── model3.py # Improved model
├── yolo.py # YOLO training
├── yolo_test.py # YOLO testing
├── data.yaml # YOLO configuration
│
├── runs/ # YOLO outputs (ignored in git)
├── object_detection_dataset/ # Dataset (ignored)
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

---

## **▶️ Run the Application**
```bash
Streamlit run present.py
```

---

## **🧪 How It Works**
1. Upload an image
2. CNN model classifies it as Bird or Drone
3. YOLO model detects objects and draws bounding boxes
4. Results are displayed in the Streamlit UI

---

## **⚠️ Limitations**
1. Model may misclassify unseen objects (e.g., trees, humans)
2. Detection accuracy depends on dataset quality
3. YOLO trained with limited epochs due to computational constraints

---

## **🧠 Technologies Used**
 Python
 TensorFlow / Keras
 Ultralytics YOLOv8
 Streamlit
 OpenCV
 NumPy
