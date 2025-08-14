import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = "AIzaSyARqUGRLuL9U2DjBs1C8JkUUAUtGB3fGPU"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
