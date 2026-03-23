from src.api.main_api import app
import uvicorn

uvicorn.run("src.api.main_api:app", host="0.0.0.0", port=8000)