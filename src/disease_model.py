import os
import warnings
import numpy as np
from PIL import Image
from keras.models import load_model
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "CNN_model.keras"
)

class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Lazy-loaded globals
_model = None
_embedding_model = None


def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))

    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def get_model():
    global _model

    if _model is None:
        print("Loading CNN model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = load_model(MODEL_PATH)

        print("CNN model loaded successfully")

    return _model


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")

        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully")

    return _embedding_model


def predict_disease(image_path):
    model = get_model()

    img = process_image(image_path)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction, axis=1)[0]

    return class_labels[predicted_index]


def encode_output(text):
    embedding_model = get_embedding_model()

    return embedding_model.encode(
        text,
        convert_to_tensor=True,
        show_progress_bar=False,
    )