from sentence_transformers import SentenceTransformer
import faiss
import json
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INDEX_PATH = os.path.join(ROOT_DIR, 'data', 'information.index')
STRUCTURED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'structured_data.json')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_vector_db():
    try:
        return faiss.read_index(INDEX_PATH)
    except Exception as e:
        print(f"Failed to load vector database: {e}")
        return None


def search_faiss(answer_query):
    vector_db = get_vector_db()
    if vector_db is None:
        return "Sorry, no relevant information found."

    emb = embedding_model.encode(answer_query)
    emb = emb.reshape(1, -1).astype('float32')

    _, indices = vector_db.search(emb, k=1)
    index = indices[0][0]

    with open(STRUCTURED_DATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if index != -1 and index < len(metadata):
        return metadata[index]

    return "Sorry, no relevant information found."


def run_rag_pipeline(answer_query):
    return search_faiss(answer_query)

    

