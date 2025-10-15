from fastapi import FastAPI, Request
from app.line_service import line_webhook

app = FastAPI()

@app.post("/line_webhook")
async def webhook(request: Request):
    return await line_webhook(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)