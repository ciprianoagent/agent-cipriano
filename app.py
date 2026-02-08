from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from cipriano import get_agent, logger
from langchain_core.messages import HumanMessage
import asyncio

app = FastAPI(title="GSurf IA Assistant")
templates = Jinja2Templates(directory="templates")

class RequestData(BaseModel):
    pergunta: str
    imagem: Optional[str] = None
    session_id: str

@app.post("/chat")
async def chat_endpoint(payload: RequestData):
    async def generate():
        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": payload.session_id}}
            content = [{"type": "text", "text": payload.pergunta}]
            if payload.imagem:
                content.append({"type": "image_url", "image_url": {"url": payload.imagem}})
            
            async for event in agent.astream_events({"messages": [HumanMessage(content=content)]}, config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk: yield chunk
        except Exception as e:
            logger.error(f"Erro no Stream: {e}")
            yield "⚠️ Falha na conexão com o motor de IA."

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})