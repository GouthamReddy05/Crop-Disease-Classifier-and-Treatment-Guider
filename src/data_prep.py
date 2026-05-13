# os.environ["USE_TF"] = "0"   # Disable TensorFlow imports
# os.environ["USE_TORCH"] = "1"  # Force PyTorch


from sentence_transformers import SentenceTransformer
import json
import os
import faiss
import numpy as np
from tqdm import tqdm

# source myenv/bin/activate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'data.json')
INDEX_PATH = os.path.join(ROOT_DIR, 'data', 'information.index')
STRUCTURED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'structured_data.json')

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
metadata = []

def dict_to_text(disease_dict):
    return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in disease_dict.items())


for dict in data:
     text = dict_to_text(dict)
     texts.append(text)


print(texts)
     


embeddings = model.encode(texts, show_progress_bar=True)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

for emb in tqdm(embeddings, desc='Indexing texts'):
    index.add(np.array([emb], dtype=np.float32))

for i, text in enumerate(texts, start=1):
    metadata.append({
        str(i): text
    })

print(f"✅ Indexed {len(metadata)} texts.")

faiss.write_index(index, INDEX_PATH)

with open(STRUCTURED_DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
