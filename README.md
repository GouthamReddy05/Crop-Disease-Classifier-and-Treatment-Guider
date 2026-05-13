# Crop-Disease-Classifier-and-TreatmentGuider

## Project Description

**Crop Disease Classifier and Treatment Guider**

• **Technologies**: Python, TensorFlow, Keras, FastAPI, Uvicorn, Pandas, Sentence Transformers, RAG, FAISS
• **Features**: Built a machine learning–driven crop disease classification and advisory system using CNN Model
• **AI Pipeline**: Implemented a Retrieval-Augmented Generation (RAG) pipeline using FAISS and Sentence Transformers for intelligent treatment guidance
• **Interface**: Delivered context-aware disease management recommendations via a modern FastAPI-based web interface with async endpoints
• **Architecture**: Modular design with organized source code, pre-trained models, and vectorized knowledge base for scalable deployment

## Project Structure

- `app.py` - Root entrypoint for running the FastAPI app
- `src/` - Application source folder
  - `src/app.py` - FastAPI application factory and routes
  - `src/disease_model.py` - Image preprocessing and disease prediction
  - `src/rag.py` - RAG / FAISS search helper for structured information
  - `src/data_prep.py` - Data preparation and vector index creation script
- `data/` - Dataset and precomputed vector store
  - `data/data.json`
  - `data/structured_data.json`
  - `data/information.index`
- `models/` - Saved model artifacts
  - `models/CNN_model.keras`
- `notebooks/` - Jupyter notebooks
- `templates/` - HTML templates
- `uploads/` - Uploaded image storage

## Setup

1. Activate your Python environment:
   ```bash
   source myenv/bin/activate
   ```

2. Install dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Groq API key:
   ```text
   GROQ_API_KEY=your_api_key_here
   ```

## Running the app

Start the FastAPI application from the repository root:

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## Notes

- The application now uses `src/app.py` as the main FastAPI app module.
- The root `app.py` file imports `create_app()` from `src` and runs it with `uvicorn`.
- The model file has been moved to `models/CNN_model.keras`.
- Structured data and FAISS index files live in `data/`.
