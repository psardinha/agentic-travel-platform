from dotenv import load_dotenv
import os
from pathlib import Path

# Ensure .env is loaded from project root
BASE_DIR = Path(__file__).resolve().parents[2]  # src/shared -> project root
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# OPENAI and Hugging Faces API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACES_API_KEY = os.getenv("HUGGING_FACES_API_KEY")

# Specifications of Travel Info agent A2A endpoint
TRAVEL_INFO_AGENT_A2A_PORT = int(os.getenv("TRAVEL_INFO_AGENT_A2A_PORT"))
TRAVEL_INFO_AGENT_SERVLET_CTX_PATH = os.getenv("TRAVEL_INFO_AGENT_SERVLET_CTX_PATH")
TRAVEL_INFO_AGENT_URL = f"http://localhost:{TRAVEL_INFO_AGENT_A2A_PORT}{TRAVEL_INFO_AGENT_SERVLET_CTX_PATH}"

# Specification of Weather Forecast MCP server endpoint
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT"))
MCP_SERVER_SERVLET_CTX_PATH = os.getenv("MCP_SERVER_SERVLET_CTX_PATH")
MCP_SERVER_URL = f"http://localhost:{MCP_SERVER_PORT}{MCP_SERVER_SERVLET_CTX_PATH}"

# Maximum number of chunck documents retrieved from vectorial database to pass to LLM for synthesis
MAX_DOCS_PASSED_TO_LLM = int(os.getenv("MAX_DOCS_PASSED_TO_LLM")) 

# SQLAlchemy-compatible URL for a file-based embedded database
DB_URI = os.getenv("DB_URI")

if __name__ == "__main__":
  print(f"MCP_SERVER_PORT = {MCP_SERVER_PORT}")
  print(f"MCP_SERVER_URL = {MCP_SERVER_URL}")