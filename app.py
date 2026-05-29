import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Page setup
st.set_page_config(
    page_title="Pneumonia CT Classification",
    page_icon="🫁",
    layout="centered"
)

class_names = ["Normal", "Pneumonia"]

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load trained model
@st.cache_resource
def load_trained_model():
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(
        torch.load("pneumonia_resnet18_model.pth", map_location=device)
    )

    model = model.to(device)
    model.eval()
    return model

model = load_trained_model()

# Prediction function
def predict_image(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    prediction = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100

    return prediction, confidence_score

# App interface
st.title("Pneumonia CT Image Classification")

st.write(
    "Upload a lung CT scan image. The model will predict whether the image is Normal or Pneumonia."
)

uploaded_file = st.file_uploader(
    "Upload CT image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded CT Image", use_container_width=True)

    if st.button("Predict"):
        prediction, confidence = predict_image(image)

        st.subheader("Prediction Result")

        if prediction == "Pneumonia":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.write(f"Confidence: **{confidence:.2f}%**")
