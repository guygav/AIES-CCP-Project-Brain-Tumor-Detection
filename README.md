# 🧠 Brain Tumor Detection using CNN & VGG-19

**📘 Course:** CT-361 – Artificial Intelligence & Expert System  
**🏛️ University:** NED University of Engineering and Technology  
**📅 Complex Computing Project**

---

## 📺 Complete Project Video Explanation
**Watch the full project walkthrough and explanation here:**
🔗 [YouTube Playlist](https://youtube.com/playlist?list=PLQB3mO-VhKGJBIx6x7nNTTCT3SO8juC7P&si=pAQQuLww_9qR80Qn)

---

## 👥 Team Members and Roles

| Member No. | Roll No.   | Name           | Responsibilities |
|------------|------------|----------------|------------------|
|   **1**    | **CR-001** | Mehak Ejaz     | 📌 Introduction, Dataset Collection & Exploratory Data Analysis |
|   **2**    | **CR-026** | Ayesha Majid   | 🧪 Data Preprocessing, Data Augmentation & Data Splitting |
|   **3**    | **CR-002** | Anum Mateen    | 🧠 CNN Model Architecture, Training, Evaluation & Performance Analysis |
|   **4**    | **CR-003** | Hafsa Usman    | 🌐 Streamlit Application, Software Tools, Final Integration & Deployment |

---

## 🧾 Section-wise Distribution

### 🔹 Member 1: Mehak Ejaz
- **Section 1:** Introduction (Background, Objective)
- **Section 2:** Literature Review
- **Section 3.1:** Data Collection
- **Section 3.2.1:** Data Organization
- **Section 3.2.2:** Exploratory Data Analysis (Class Distribution, Visualization, Image Characteristics)

### 🔹 Member 2: Ayesha Majid
- **Section 3.2.3:** Data Augmentation
- **Section 3.2.4:** Data Preparation (Resizing, Normalization, Label Encoding)
- **Section 3.2.5:** Data Splitting (Stratified Shuffle & Test Split)

### 🔹 Member 3: Anum Mateen
- **Section 3.3:** CNN Model Building (Architecture Design + Use of Pre-trained VGG19)
- **Section 5.1:** Performance Enhancement (Transfer Learning, Fine-Tuning, Freezing Layers)
- **Section 5.2:** Training Accuracy and Loss Visualization
- **Model Training & Evaluation Code**

### 🔹 Member 4: Hafsa Usman
- **Section 3.4:** Streamlit Web Application
- **Section 4.1:** Software and Tools Used
- **Section 5:** Results and Deployment
- **Section 6:** Conclusion

---

## ✅ Project Features
- Deep Learning-based Brain Tumor Classification
- VGG19-based CNN Architecture
- Preprocessing with OpenCV & Keras
- Real-time Web App using Streamlit
- Model Evaluation using Accuracy & Loss plots

---

> This project was developed as part of the CT-377 Artificial Intelligence & Expert System course at **NED University of Engineering & Technology**.

## 📌 Overview

Traditional brain tumor diagnosis through biopsy is invasive, expensive, and time-consuming. With the power of deep learning, this project uses Convolutional Neural Networks (CNNs) to classify brain MRI scans as **Tumor** or **No Tumor**, enabling fast, non-invasive diagnosis.

This project includes:
- 📷 MRI image classification using deep learning
- 🧠 Custom CNN and transfer learning models (VGG-19, ResNet50, MobileNet)
- 🌐 Deployment via a Streamlit app for real-time predictions

## 🗂️ Dataset

- **Source**: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download)
- **Classes**: `Tumor` and `No Tumor`
- **Total Images**: ~253  
  - Tumor: 155  
  - No Tumor: 98
- **Format**: JPG
- **Resized Dimensions**: `128x128` and `224x224` for different models

## 🔍 Methodology

### 1. 🔧 Data Preprocessing
- Resizing all images to fixed input size
- Normalization of pixel values to `[0, 1]`
- Label encoding: Tumor → `1`, No Tumor → `0`

### 2. 📈 Data Augmentation
- Rotation
- Horizontal/vertical flipping
- Random zoom
- Brightness and contrast adjustments
- Cropping and padding

### 3. 📊 Data Splitting
- **Train**: 70%
- **Validation**: 20%
- **Test**: 10%

## 🧠 Model Architectures

### 🔹 Custom CNN
- Multiple convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Final sigmoid layer for binary output

### 🔹 Transfer Learning Models
- **Pre-trained Architectures**:  
  - VGG-19  
  - ResNet50  
  - MobileNet

**Techniques Used**:
- Freezing early layers
- Fine-tuning deeper layers
- Transferring knowledge from ImageNet weights

## 🖥️ VGG-19 Model Details

### 🎯 Goal
Classify MRI scans to determine tumor presence using the VGG-19 architecture.

### ⚙️ Configuration
- **Architecture**: VGG-19
- **Transfer Learning**: Pre-trained on ImageNet
- **Fine-Tuning**: Applied to deeper layers for domain-specific learning

### 📈 Evaluation Metric
Accuracy = (Correct Predictions / Total Images) × 100%

### 💡 Final Results

| **Dataset**       | **Accuracy** |  
|--------------------|--------------|  
| **Validation Set** | ~88%         |  
| **Test Set**       | ~80%         |  

## 🌐 Streamlit App

An interactive web app built with Streamlit:

- Upload an MRI image
- Model predicts: ✅ Tumor or ❌ No Tumor
- Shows the image and prediction in real-time

## 📊 Overall Project Results

| Metric               | Custom CNN | VGG-19 |
|----------------------|------------|--------|
| Validation Accuracy  | 65.26%     | ~88%   |
| Test Accuracy        | 66.77%     | ~80%   |
| Validation Loss      | 0.6172     | -      |
| Test Loss            | 0.6082     | -      |

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Anum-Mateen/brain-tumor-detection.git
cd brain-tumor-detection-cnn
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```
streamlit run app.py
```
Upload an MRI image in the browser to get an instant prediction.

## 💻 Running on Google Colab

### ✅ 1. Open the Colab Notebook
```
Link: https://colab.research.google.com/
```

### 📂 2. Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```

### 📦 3. Install Required Libraries
Run the following cell at the beginning of your notebook:
```python
!pip install tensorflow keras opencv-python matplotlib seaborn streamlit scikit-learn imutils
```

### 🗃️ 4. Unzip Dataset (if in Drive)
```
import zipfile
zip_path = '/content/drive/MyDrive/BrainTumorProject/data.zip'  # Change to your path
extract_path = '/content/data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### 📊 5. Start Training or Inference
Once the dataset is unzipped and libraries installed, you can:
- Train your CNN/VGG-19 model
- Load a saved model and predict on new images

## 🛠️ Tools & Libraries Used

- Python
- TensorFlow / Keras
- OpenCV (opencv-python)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Pillow (PIL)
- Imutils
- Scikit-learn

## 📄 License

This project is licensed under the MIT License.
See the [LICENSE](https://github.com/Anum-Mateen/Brain-Tumor-Detection/blob/main/LICENSE) file for more details.

## 🙌 Acknowledgements

- Kaggle: Brain MRI Dataset
- The Cancer Imaging Archive (TCIA)
- Pre-trained models from TensorFlow/Keras Model Zoo
