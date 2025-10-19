from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from chat import chat_router
from config import limiter

# Initialize FastAPI app
app = FastAPI(title="SupportGPT")

# Configure limiter from chat module
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Include chat router
app.include_router(chat_router)
