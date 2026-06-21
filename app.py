from src import create_app
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = create_app()
print("ROOT APP IMPORTED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=5000, reload=True)
