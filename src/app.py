from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import os
import re
from dotenv import load_dotenv
import traceback
from pathlib import Path
from .disease_model import predict_disease
from .rag import run_rag_pipeline
import uuid

print("STARTING APP")

load_dotenv()

print("APP FILE LOADED")

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = ROOT_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}



# Load Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

templates = Jinja2Templates(directory=str(ROOT_DIR / 'templates'))


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return filename


def groq_llm(prompt: str, system_prompt: str | None = None) -> str:
    """Send prompt to Groq API and get the response."""
    if not groq_api_key:
        return "Error: Groq API key not configured. Please set GROQ_API_KEY in your .env file."

    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )

        if not response.ok:
            return f"Error: {response.status_code} - {response.text}"

        result = response.json()

        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "Error: Unexpected API response format."

    except Exception as e:
        print(f"Error communicating with Groq API: {e}")
        return f"Error communicating with Groq API: {str(e)}"


def create_app() -> FastAPI:
    app = FastAPI()
    print("CREATING FASTAPI APP")
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    @app.get('/', response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main chat interface."""
        template = templates.get_template('index.html')
        html_content = template.render(request=request)
        return HTMLResponse(content=html_content, status_code=200)

    @app.post('/chat')
    async def chat(payload: dict):
        """Handle text-only chat messages."""
        user_message = payload.get('message', '').strip()

        if not user_message:
            raise HTTPException(status_code=400, detail='No message provided')

        system_prompt = (
            "You are a helpful AI medical assistant. Provide informative, accurate medical information "
            "while always recommending users consult healthcare professionals for serious concerns. Be empathetic "
            "and helpful. If you're unsure about something, say so. Format your responses clearly with line breaks where appropriate."
        )

        llm_result = groq_llm(user_message, system_prompt)

        return {'reply': llm_result, 'status': 'success'}

    @app.post('/process-image')
    async def process_image_route(
        image: UploadFile = File(...),
        message: str | None = Form(None)
    ):
        """Handle image uploads and return prediction + LLM response."""
        if not image.filename:
            raise HTTPException(status_code=400, detail='No file selected')

        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail=f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            )

        filename = f"{uuid.uuid4()}_{secure_filename(image.filename)}"
        image_path = UPLOAD_FOLDER / filename

        try:
            content = await image.read()
            image_path.write_bytes(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error saving upload: {e}')

        print(f"✅ Image uploaded successfully: {image_path}")
        print(f"📁 File size: {image_path.stat().st_size} bytes")
        print(f"💬 User message: {message}")

        try:
            # _ = process_image(str(image_path))
            disease_prediction = predict_disease(str(image_path))
            # _ = encode_output(disease_prediction)
            rag_result = run_rag_pipeline(disease_prediction)

            llm_response = groq_llm(
                f"Based on the disease prediction '{disease_prediction}' and this context: {rag_result}, provide relevant information.",
                system_prompt="You are a helpful AI assistant providing information based on disease predictions and also add extra information for better understanding instead of directly repeating the context."
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f'Error processing image: {e}')

        return {
            'reply': llm_response,
            'image_path': str(image_path),
            'filename': filename,
            'file_size': image_path.stat().st_size,
            'status': 'success'
        }

    @app.get('/health')
    async def health_check():
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'groq_api_configured': bool(groq_api_key),
            'upload_folder_exists': UPLOAD_FOLDER.exists()
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse({'error': exc.detail, 'status': 'error'}, status_code=exc.status_code)

    return app


app = create_app()
