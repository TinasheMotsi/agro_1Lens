# ==================== AGROLENS WEB APP ====================
import streamlit as st
import torch
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="AgroLens - Crop Disease Detector",
    page_icon="Leaf",
    layout="centered"
)


# ----------------- Load Model Once -----------------
@st.cache_resource
def load_model():
    model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model


processor, model = load_model()

# ----------------- Classes & Treatments -----------------
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

treatments = {
    'Early blight': 'Apply copper or chlorothalonil fungicide. Remove lower leaves. Improve air circulation.',
    'Late blight': 'Urgent! Use metalaxyl/mancozeb. Destroy infected plants. Do not compost.',
    'Bacterial spot': 'Copper-based sprays. Avoid overhead watering. Rotate crops.',
    'Leaf Mold': 'Increase ventilation. Apply fungicides. Keep foliage dry.',
    'healthy': 'Plant is healthy! Keep up good farming practices.',
    'Powdery mildew': 'Sulfur or potassium bicarbonate sprays. Prune for airflow.',
    'Common rust': 'Fungicides with triazoles. Plant resistant varieties.',
    'Northern Leaf Blight': 'Rotate crops. Use resistant hybrids. Apply strobilurin fungicides.',
}


# ----------------- Image Enhancement -----------------
def enhance_image(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img


# ----------------- Prediction Function -----------------
def predict(image):
    image = enhance_image(image)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        top_prob, top_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top_idx[0][i].item()
        prob = top_prob[0][i].item() * 100
        disease_class = classes[idx]
        parts = disease_class.split("___")
        crop = parts[0].replace("_(including_sour)", "").replace("_(maize)", "").replace("_bell,",
                                                                                         " Bell Pepper").strip()
        status = parts[1].replace("_", " ").split("(")[0].strip()
        results.append((crop, status, prob))

    return results


# ----------------- Streamlit UI -----------------
st.title("Leaf AgroLens")
st.markdown("### Instant Crop Disease Diagnosis from Leaf Photos")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "jpeg", "png"])
with col2:
    st.image(
        "https://img.freepik.com/free-photo/closeup-shot-green-tomatoes-growing-branch-with-leaves_181624-24110.jpg?w=740",
        caption="Example leaf", use_column_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    with st.spinner("Analyzing with AI..."):
        results = predict(image)

    st.success("Diagnosis Complete!")

    top_crop, top_disease, top_conf = results[0]
    st.metric("Most Likely Disease", f"{top_crop} → {top_disease}", f"{top_conf:.1f}% confidence")

    st.subheader("Top 3 Predictions")
    for crop, disease, conf in results:
        status = "Healthy" if "healthy" in disease.lower() else "Diseased"
        st.write(f"- **{crop}** → **{disease}** ({conf:.1f}%) {'Healthy' if status == 'Healthy' else 'Warning'}")

    tip = treatments.get(top_disease.split()[0] + " " + " ".join(top_disease.split()[1:]),
                         "Monitor plant. Consult local agriculture expert for best treatment.")
    st.info(f"Recommended Action: {tip}")

st.markdown("---")
st.caption("AgroLens v3 • Powered by AI • Built with Streamlit")