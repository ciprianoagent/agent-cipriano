import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# Importamos a função que vamos criar no cipriano.py
from cipriano import executar_agente

app = FastAPI()

# Configura a pasta de templates para o HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def pagina_inicial(request: Request):
    """Renderiza a página inicial do site"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(pergunta: str = Form(...)):
    """Recebe a pergunta do site e retorna a resposta da IA"""
    try:
        resposta = executar_agente(pergunta)
        return {"resposta": resposta}
    except Exception as e:
        return {"resposta": f"Erro no agente: {str(e)}"}

if __name__ == "__main__":
    # Garante que o app rode na porta correta do Render
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)