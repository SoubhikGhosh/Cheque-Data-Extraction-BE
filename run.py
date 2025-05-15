import uvicorn
from app.core.config import settings # To access host/port from settings if defined

if __name__ == "__main__":
    # You can add host and port to your .env and Settings if you want to configure them
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True, # Disable reload for production
        log_level=settings.LOG_LEVEL.lower() # Sync uvicorn log level with app
    )