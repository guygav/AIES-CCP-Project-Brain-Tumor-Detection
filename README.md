# 🧠 Brain Tumor Detection using CNN & VGG-16

## 📌 Overview

Traditional brain tumor diagnosis through biopsy is invasive, expensive, and time-consuming. With the power of deep learning, this project uses Convolutional Neural Networks (CNNs) to classify brain MRI scans as **Tumor** or **No Tumor**, enabling fast, non-invasive diagnosis.

This project includes:
- 📷 MRI image classification using deep learning
- 🧠 Custom CNN and transfer learning models (VGG-16, ResNet50, MobileNet)
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
  - VGG-16  
  - ResNet50  
  - MobileNet

**Techniques Used**:
- Freezing early layers
- Fine-tuning deeper layers
- Transferring knowledge from ImageNet weights

## 🖥️ VGG-16 Model Details

### 🎯 Goal
Classify MRI scans to determine tumor presence using the VGG-16 architecture.

### ⚙️ Configuration
- **Architecture**: VGG-16
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

| Metric               | Custom CNN | VGG-16 |
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
Link: (Add your notebook link here)
```

### 📂 2. Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```

### 📦 3. Install Required Libraries
```
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
- Train your CNN/VGG-16 model
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
See the LICENSE(https://github.com/Anum-Mateen/Brain-Tumor-Detection/blob/main/LICENSE) file for more details.

## 🙌 Acknowledgements

- Kaggle: Brain MRI Dataset
- The Cancer Imaging Archive (TCIA)
- Pre-trained models from TensorFlow/Keras Model Zoo
