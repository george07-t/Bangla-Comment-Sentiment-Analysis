# 📝 Bangla Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/Language-Python-blue.svg) ![NLP](https://img.shields.io/badge/Domain-NLP-green.svg)

## 🚀 Project Overview

This project focuses on sentiment analysis of Bangla text data. The goal is to classify comments into positive and negative categories using advanced deep learning techniques. The model leverages a state-of-the-art Transformer-based architecture, specifically fine-tuned for Bangla language processing.

### 🎯 Key Features
- **Text Preprocessing**: Includes custom Bangla text processing pipeline with tokenization, stopword removal, and text normalization.
- **Model Architecture**: Utilizes a Transformer-based model, specifically fine-tuned for the Bangla language to handle text efficiently and effectively.
- **Performance Metrics**: Achieves exceptionally high accuracy, F1-Score, and AUC, demonstrating robust classification performance.

## 📊 Results

### Training and Validation Loss
![Training and Validation Loss](https://github.com/user-attachments/assets/e3773519-2039-4adf-acd4-0db975caf1b0)

The training loss decreases steadily while the validation loss exhibits some fluctuations, indicating a strong learning process with minimal overfitting. The model generalizes well on the validation set.

### ROC Curve
![ROC Curve](https://github.com/user-attachments/assets/ad028768-d59e-457b-a15b-8648cfc23bb5)

The ROC curve shows an AUC of 1.00, indicating near-perfect classification performance, which reflects the model's ability to distinguish between positive and negative sentiments effectively.

### Model Performance by Epoch
| Epoch | Training Loss | Validation Loss | Accuracy | F1 | Precision | Recall | MCC  |
|-------|---------------|-----------------|----------|----|-----------|--------|------|
| 1     | 0.033700      | 0.043993        | 0.992062 | 0.940988 | 0.890578  | 0.997447 | 0.938476 |
| 2     | 0.061900      | 0.023995        | 0.995194 | 0.963299 | 0.934400  | 0.994043 | 0.961257 |
| 3     | 0.000700      | 0.022209        | 0.995518 | 0.965688 | 0.938907  | 0.994043 | 0.963742 |
| 4     | 0.000000      | 0.023287        | 0.995680 | 0.966942 | 0.939759  | 0.995745 | 0.965095 |
| 5     | 0.000200      | 0.020583        | 0.995896 | 0.968491 | 0.944220  | 0.994043 | 0.966663 |
| 6     | 0.000100      | 0.035792        | 0.994438 | 0.957804 | 0.923381  | 0.994894 | 0.955592 |
| 7     | 0.000100      | 0.025074        | 0.995626 | 0.966515 | 0.939711  | 0.994894 | 0.964626 |
| 8     | 0.030800      | 0.021211        | 0.996490 | 0.972701 | 0.960199  | 0.985532 | 0.970919 |
| 9     | 0.045600      | 0.021171        | 0.996382 | 0.972025 | 0.954098  | 0.990638 | 0.970287 |
| 10    | 0.000000      | 0.026539        | 0.996058 | 0.969621 | 0.948697  | 0.991489 | 0.967784 |
| 11    | 0.000000      | 0.029619        | 0.996112 | 0.970025 | 0.949470  | 0.991489 | 0.968207 |
| 12    | 0.000100      | 0.025535        | 0.996328 | 0.971524 | 0.956307  | 0.987234 | 0.969703 |
| 13    | 0.000000      | 0.023126        | 0.996166 | 0.969979 | 0.963866  | 0.976170 | 0.967953 |

**Final Validation Results**:
- **Accuracy**: 99.64%
- **F1-Score**: 0.9727
- **Precision**: 0.9601
- **Recall**: 0.9855
- **MCC**: 0.9709

## 📂 Project Structure

- `data/`: Contains the dataset files used for training and validation.
- `notebooks/`: Jupyter notebooks with step-by-step code and analysis.
- `models/`: Saved model files and checkpoints.
- `src/`: Source code for data preprocessing, model building, and evaluation.
- `results/`: Evaluation metrics, plots, and logs.

## 📚 Dataset Description

The dataset used in this project consists of Bangla text comments labeled as either positive or negative. The dataset is imbalanced, with a higher number of negative comments compared to positive ones.

### Dataset Details

- **Positive Comments**: 1,267
- **Negative Comments**: 7,214

The comments are collected from various sources, primarily focusing on social media and user-generated content platforms where Bangla is predominantly used.

### How to Find the Data

The dataset can be found on Kaggle and is available for download at the following link:

[Kaggle Dataset Link](https://www.kaggle.com/your-dataset-link)

Once you download the dataset from Kaggle, place the relevant files in the `data/` directory of this repository. The dataset should include:

- `train.csv`: The training dataset with comments and their corresponding sentiment labels.
- `test.csv`: The test dataset used for evaluating model performance.

If you wish to use your own dataset, place it in the `data/` directory and ensure it follows the same structure as the provided files:

- **Columns**: `comment`, `label`
  - `comment`: The text of the comment in Bangla.
  - `label`: The sentiment label, where `1` represents positive sentiment and `0` represents negative sentiment.

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

- **Positive Comments**: 5,731
- **Negative Comments**: 86,855

The dataset is significantly imbalanced, which could affect model performance. Consider using techniques like SMOTE or class weighting to address this.

### Recommendations
- **Class Imbalance**: Address the imbalance in the dataset to improve the model's precision for the minority class.

## 🤝 **Contributing**

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements, bug fixes, or additional features to suggest.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

   
