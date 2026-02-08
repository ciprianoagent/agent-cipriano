import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # Opcional, mas boa prática
from pydantic import BaseModel
from typing import Optional

# Importamos a função do arquivo cipriano.py
from cipriano import executar_agente

app = FastAPI(title="GSurf IA Assistant")

# Configura diretório de templates
# Certifique-se de ter uma pasta chamada 'templates' e o index.html dentro dela
templates = Jinja2Templates(directory="templates")

# Modelo de Dados (Protocolo de Comunicação Front-Back)
class RequestData(BaseModel):
    pergunta: str
    imagem: Optional[str] = None
    session_id: str  # CRUCIAL: Identificador único do usuário para a memória

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renderiza a interface de chat."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: RequestData):
    """
    Recebe a mensagem + sessão + imagem (opcional)
    e invoca o agente Cipriano.
    """
    try:
        # Chama o agente passando o ID da sessão para manter o contexto
        resposta_texto = executar_agente(
            mensagem_usuario=payload.pergunta, 
            imagem_b64=payload.imagem,
            session_id=payload.session_id
        )
        
        return {"resposta": resposta_texto}

    except Exception as e:
        print(f"Erro no Endpoint /chat: {e}")
        # Retorna erro amigável para o front não quebrar
        return {"resposta": f"⚠️ **Erro de Sistema:** Não foi possível processar sua solicitação. Detalhe: {str(e)}"}

if __name__ == "__main__":
    # Configuração para rodar no Render/Heroku ou Local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)