# Pneumonia Classification System

This project is a deep learning-based web application that classifies lung CT scan images as **Normal** or **Pneumonia**.

The model was trained using a pretrained ResNet-18 architecture in PyTorch and integrated into a Streamlit web application for single-image prediction.

## Features

- Upload a lung CT scan image
- Predict whether the image is Normal or Pneumonia
- Display prediction confidence score
- Simple Streamlit-based user interface

## Model Details

- Architecture: ResNet-18
- Framework: PyTorch
- Classes: Normal, Pneumonia
- Validation Accuracy: 99.36%
- Test Accuracy: 99.36%
- Pneumonia Recall: 1.00

## Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow
- Google Colab
- VS Code

## Project Files

- `pneumonia_ct_classification.ipynb` - Google Colab notebook containing data loading, preprocessing, model training, evaluation, and model saving steps.
- `app.py` - Streamlit web application for uploading CT images and predicting Normal or Pneumonia.
- `pneumonia_resnet18_model.pth` - Saved trained PyTorch model weights.
- `requirements.txt` - Required Python packages.

## How to Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python3 -m streamlit run app.py
```

## Dataset

The dataset used for this project was obtained from Kaggle and contains lung CT scan images categorized into Normal and Pneumonia classes.

The dataset is not included in this repository due to file size and licensing limitations.

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis.