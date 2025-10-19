from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

from openai_client import OpenAIClient
from storage import Storage

# Load environment variables
load_dotenv()

# Initialize IP limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize OpenAI client
openai_client = OpenAIClient()

# Initialize conversation "database"
storage = Storage(db_path="storage/records")
