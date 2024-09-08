# 📝 Bangla Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/Language-Python-blue.svg) ![NLP](https://img.shields.io/badge/Domain-NLP-green.svg)

## 🚀 Project Overview

This project focuses on sentiment analysis of Bangla text data. The goal is to classify comments into positive and negative categories using advanced machine learning techniques. The model leverages various NLP preprocessing steps and is trained using a deep learning model with custom layers tailored for Bangla language processing.

### 🎯 Key Features
- **Text Preprocessing**: Custom Bangla text processing pipeline including tokenization, stopword removal, and text normalization.
- **Model Architecture**: Utilizes deep learning architecture, possibly involving LSTM or Transformer-based models, to handle Bangla text efficiently.
- **Performance Metrics**: High accuracy, F1-Score, and AUC, demonstrating robust classification performance.

## 📊 Results

### Training and Validation Loss
![image](https://github.com/user-attachments/assets/aa2fc9a5-45c6-47f1-ae3f-c44e120679d2)



The training loss decreases significantly while the validation loss remains relatively stable, indicating a good learning process but potential overfitting in later epochs.

### ROC Curve
![image](https://github.com/user-attachments/assets/1f07807d-9d1a-4678-aa55-86c60a11c90d)

The ROC curve shows an AUC of 1.00, indicating near-perfect classification, though this should be interpreted cautiously given the class imbalance.

### Model Performance by Epoch
| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score | Precision | Recall | MCC  |
|-------|---------------|-----------------|----------|----------|-----------|--------|------|
| 1     | 0.0633        | 0.1056          | 95.58%   | 0.8691   | 0.7955    | 0.9577 | 0.8480 |
| 2     | 0.0128        | 0.1132          | 97.70%   | 0.9295   | 0.8771    | 0.9885 | 0.9181 |
| 3     | 0.0479        | 0.1106          | 97.76%   | 0.9309   | 0.8828    | 0.9846 | 0.9195 |

**Final Validation Results**:
- **Accuracy**: 97.76%
- **F1-Score**: 0.9309
- **Precision**: 0.8828
- **Recall**: 0.9846
- **MCC**: 0.9195

## 📂 Project Structure

- `data/`: Contains the dataset files used for training and validation.
- `notebooks/`: Jupyter notebooks with step-by-step code and analysis.
- `models/`: Saved model files and checkpoints.
- `src/`: Source code for data preprocessing, model building, and evaluation.
- `results/`: Evaluation metrics, plots, and logs.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bangla-sentiment-analysis.git
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook notebooks/bangla-sentiment-analysis.ipynb
   
## 🔍 **Analysis**

### Class Distribution

- **Positive Comments**: 1,267
- **Negative Comments**: 7,214

The dataset is significantly imbalanced, which could affect model performance. Consider using techniques like SMOTE or class weighting to address this.

### Recommendations

- **Overfitting**: The model shows signs of overfitting; consider using regularization techniques or early stopping.
- **Class Imbalance**: Address the imbalance in the dataset to improve the model's precision for the minority class.

## 🤝 **Contributing**

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements, bug fixes, or additional features to suggest.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

   
