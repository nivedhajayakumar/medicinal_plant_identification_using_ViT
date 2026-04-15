# 🌿 Medicinal Plant Classification using Vision Transformer (ViT)

A deep learning project for classifying **40 species of medicinal plants** using a fine-tuned Vision Transformer (ViT) model — achieving **99.24% validation accuracy**.

---

## 📌 Overview

This project fine-tunes Google's `vit-base-patch16-224` pretrained model on a custom dataset of Indian medicinal plant images. The model learns to identify plant species from leaf images, which can be useful in healthcare, agriculture, and botanical research.

---

## 🗂️ Dataset

- **Source:** Indian Medicinal Leaves Image Dataset (stored in Google Drive)
- **Total Images:** 5,945
- **Classes:** 40 medicinal plant species
- **Split:** 80% training / 20% validation

### Plant Classes

| | | | |
|---|---|---|---|
| Aloevera | Amla | Amruta_Balli | Arali |
| Ashoka | Ashwagandha | Avacado | Bamboo |
| Basale | Betel | Betel_Nut | Brahmi |
| Castor | Curry_Leaf | Doddapatre | Ekka |
| Ganike | Gauva | Geranium | Henna |
| Hibiscus | Honge | Insulin | Jasmine |
| Lemon | Lemon_grass | Mango | Mint |
| Nagadali | Neem | Nithyapushpa | Nooni |
| Pappaya | Pepper | Pomegranate | Raktachandini |
| Rose | Sapota | Tulasi | Wood_sorel |

---

## 🧠 Model Architecture

- **Base Model:** `google/vit-base-patch16-224` (Vision Transformer)
- **Framework:** PyTorch + HuggingFace Transformers
- **Classifier Head:** Modified final linear layer → 40 output classes
- **Input Size:** 224 × 224 RGB images

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 5e-5 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Device | GPU (CUDA) |

### Training Progress

| Epoch | Loss |
|-------|------|
| 1 | 1.3139 |
| 2 | 0.0934 |
| 3 | 0.0293 |
| 5 | 0.0102 |
| 10 | 0.0019 |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **99.24%** |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision transformers pillow huggingface_hub
```

### Clone the Repository

```bash
git clone https://github.com/your-username/medicinal-plant-classifier.git
cd medicinal-plant-classifier
```

### Dataset Setup

1. Download the medicinal plant dataset and place it in your working directory (or Google Drive if using Colab):
```
Medicinal plant dataset/
├── Aloevera/
├── Amla/
├── Tulasi/
└── ... (40 classes total)
```

2. Update `data_dir` in the notebook to point to your dataset path.

### Training

Run the notebook in Google Colab (GPU recommended):

```python
# Set data directory
data_dir = '/path/to/Medicinal plant dataset'

# Training runs for 10 epochs (~33 minutes on GPU)
```

### Inference

```python
from PIL import Image
from torchvision import transforms
import torch

# Load saved model
model.load_state_dict(torch.load('medplant.pth'))
model.eval()

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image = Image.open('your_plant_image.jpg')
input_tensor = preprocess(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor).logits
    predicted_class = output.argmax(dim=1).item()
    print(f"Predicted class index: {predicted_class}")
```

---

## 💾 Saved Model

The trained model weights are saved as `medplant.pth` and can be loaded directly for inference without retraining.

---

## 📁 Project Structure

```
medicinal-plant-classifier/
├── medicinal_plant_classification.ipynb   # Main training notebook
├── medplant.pth                           # Saved model weights
└── README.md
```

---

## 🛠️ Tech Stack

- Python 3.10
- PyTorch
- HuggingFace Transformers
- TorchVision
- Google Colab (with GPU)
- Google Drive (dataset storage)

---

## 📄 License

This project is for academic and research purposes.

---

## 🙏 Acknowledgements

- [Google ViT](https://huggingface.co/google/vit-base-patch16-224) pretrained model via HuggingFace
- Indian Medicinal Leaves Image Dataset
