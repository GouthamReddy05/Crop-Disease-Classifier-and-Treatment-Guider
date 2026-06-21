import os
import warnings
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
from keras.models import load_model


warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'CNN_model.keras')


class_labels = [
 'Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy'
]



def process_image(inp_img):

    img = Image.open(inp_img).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img).astype(np.float32)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)     ## (1, 256, 256, 3)

    return img


model = None
emb_model = None


def get_model():
    global model
    if model is None:
        print("Loading CNN model...")
        model = load_model(MODEL_PATH)
    return model


def get_embedding_model():
    global emb_model
    if emb_model is None:
        print("Loading embedding model...")
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return emb_model

def predict_disease(image_path):
    img = process_image(image_path)
    model = get_model()
    pred = model.predict(img)
    predicted_index = np.argmax(pred, axis=1)
    return f"{class_labels[int(predicted_index)]}"

# print(predict_disease("sample.JPG"))




def encode_output(text):
    emb_model = get_embedding_model()
    return emb_model.encode(text, convert_to_tensor=True, show_progress_bar=False)