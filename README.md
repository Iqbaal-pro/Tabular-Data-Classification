# Tabular-Data-Classification
Binary classification of rice grains using a feedforward neural network built with PyTorch. Covers full ML pipeline: preprocessing, custom Dataset, training loop, evaluation, and inference.

A Tebular data classification project built with PyTorch that classifies rice grains using physical measurements. This project covers the full machine learning pipeline — from data preprocessing to model training, evaluation, and inference.

---

## 📌 Project Overview

This project trains a neural network to classify rice grains into two categories based on physical features like area, perimeter, eccentricity, and more. It uses the [Rice Classification Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification).

---

## 🧠 Model Architecture

```
Input Layer  →  Linear(10, 10)  →  ReLU  →  Linear(10, 1)  →  Sigmoid
```

- **Input:** 10 numerical features per rice grain
- **Output:** Binary classification (0 or 1)
- **Loss Function:** Binary Cross-Entropy (BCELoss)
- **Optimizer:** Adam (lr = 1e-3)

---

## 📁 Project Structure

```
├── First_PyTorch_Project.ipynb   # Main notebook
├── riceClassification.csv        # Dataset (not included, see below)
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle - Rice Type Classification](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)
- **Features:** Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation
- **Target:** Class (binary — 0 or 1)

> Download the dataset from Kaggle and place `riceClassification.csv` in the same directory as the notebook.

---

## ⚙️ Setup & Installation

```bash
pip install torch torchvision torchsummary scikit-learn pandas numpy matplotlib
```

Then open the notebook:

```bash
jupyter notebook First_PyTorch_Project.ipynb
```

Or run it directly on [Google Colab](https://colab.research.google.com/).

---

## 🏋️ Training

The model is trained for **10 epochs** with a **batch size of 32**.

The dataset is split as follows:
- **70%** Training
- **15%** Validation
- **15%** Test

Training and validation loss/accuracy are tracked and plotted after training.

---

## 📈 Results

| Split      | Accuracy  |
|------------|-----------|
| Training   | ~XX%      |
| Validation | ~XX%      |
| Test       | ~XX%      |

> Replace XX% with your actual results after running the notebook.

---

## 🔍 Sample Inference

After training, the model can predict the class of a new rice grain given its physical measurements:

```python
my_prediction = model(torch.tensor([area, MajorAxisLength, MinorAxisLength, ...], dtype=torch.float32))
print("Predicted class:", round(my_prediction.item()))
```

---

## 📚 What I Learned

- Building a custom PyTorch `Dataset` and `DataLoader`
- Designing and training a feedforward neural network
- Normalizing features for stable training
- Monitoring overfitting with a validation loop
- Visualizing training curves with Matplotlib

---

## 🙋 Author

**[Iqbaal Meedin]**
- GitHub: [iqbaal-pro](https://github.com/iqbaal-pro)


---
