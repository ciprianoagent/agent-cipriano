import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ======================================================
# TOOL — Web Search (OSINT)
# ======================================================
@tool
def search_web(query: str) -> str:
    """Busca informações públicas e atuais na internet (OSINT)."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "TAVILY_API_KEY não configurada no Render."
    
    search = TavilySearchResults(max_results=3) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# AGENT LOGIC - CIPRIANO
# ======================================================
system_message = """
Você é Cipriano, um agente estratégico de Inteligência e Segurança,
com a presença, autoridade e frieza calculada de Don Corleone.
- Seja direto e use tom formal/imponente.
- Atue apenas em contextos legais e éticos.
"""

def executar_agente(mensagem_usuario: str):
    """Função de interface com o app.py"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "Erro: GOOGLE_API_KEY não encontrada no ambiente."
    
    # AJUSTE DO MODELO: Usando o nome completo para evitar o erro 404
    # 'gemini-1.5-flash-latest' costuma ser o mais estável para a v1beta
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0,
        api_key=api_key
    )
    
    agent = create_react_agent(
        model=model, 
        tools=tools, 
        prompt=system_message
    )
    
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "thread-1"}}
    
    try:
        resultado = agent.invoke(inputs, config)
        
        # AJUSTE DE EXTRAÇÃO (Resolve as imagens com texto gigante):
        # Acessamos a última mensagem e pegamos apenas o campo .content
        ultima_mensagem = resultado["messages"][-1]
        
        if hasattr(ultima_mensagem, 'content'):
            return ultima_mensagem.content
        return str(ultima_mensagem)
        
    except Exception as e:
        # Se o modelo 'latest' falhar, tentamos o nome simples como fallback
        return f"Cipriano informa: Falha na requisição. Verifique as credenciais. (Erro: {str(e)})"