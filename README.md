# Potato Leaf Disease Detection 🍂

This is a **Streamlit-based web application** for detecting **potato leaf diseases** using a trained deep learning model.

## 🚀 Features
- Upload an image of a potato leaf.
- Predict whether the leaf is **healthy, affected by Early Blight, or affected by Late Blight**.
- Uses a **TensorFlow/Keras** trained model for predictions.
- Simple and interactive **Streamlit UI**.

## 📌 Setup Instructions
### 1️⃣ Install Dependencies
Make sure you have Python installed. Then install the required libraries:
```bash
pip install streamlit tensorflow numpy opencv-python pillow
```

### 2️⃣ Run the Application
```bash
streamlit run app.py
```

## 📂 Project Structure
```
📦 PotatoLeafDiseaseDetection
├── 📄 app.py                # Streamlit application script
├── 📄 trained_plant_disease_model.keras  # Pre-trained Keras model
├── 📄 README.md             # Project documentation (this file)
```

## 🖼️ Usage
1. Run the app using `streamlit run app.py`.
2. Upload an image of a **potato leaf**.
3. Click **"Predict Disease 🩺"**.
4. Get instant results with **disease name & confidence score**.

## 🏆 Model Details
- The model is trained on **Potato Leaf Disease Dataset**.
- It classifies images into:
  - **Potato___Healthy** 🌱
  - **Potato___Early_blight** 🍂
  - **Potato___Late_blight** 🍁

## 🔧 Troubleshooting
- Ensure the `trained_plant_disease_model.keras` file is in the correct directory.
- If Streamlit fails to run, try:
  ```bash
  pip install --upgrade streamlit
  ```
- For debugging, check logs with:
  ```bash
  streamlit run app.py --logger.level=debug
  ```

## 📜 License
This project is **open-source** and free to use!

---

⭐ *Developed for Agricultural Disease Detection using AI.*
