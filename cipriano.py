import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ======================================================
# TOOL — Web Search
# ======================================================
@tool
def search_web(query: str) -> str:
    """Busca informações públicas e atuais na internet (OSINT)."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "TAVILY_API_KEY não configurada no Render."
    
    # O Tavily lerá a chave automaticamente da variável de ambiente TAVILY_API_KEY
    search = TavilySearchResults(max_results=3) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# AGENT LOGIC
# ======================================================
system_message = """
Você é Cipriano, um agente estratégico de Inteligência e Segurança,
com a presença, autoridade e frieza calculada de Don Corleone.
- Seja direto e use tom formal/imponente.
- Atue apenas em contextos legais e éticos.
"""

def executar_agente(mensagem_usuario: str):
    """Função que o app.py vai chamar"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "Erro: GOOGLE_API_KEY não encontrada no ambiente do Render."
    
    # Modelo estável para produção
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0,
        api_key=api_key
    )
    
    # AJUSTE REALIZADO: Alterado 'state_modifier' para 'prompt'
    agent = create_react_agent(
        model=model, 
        tools=tools, 
        prompt=system_message
    )
    
    # Executa o grafo do agente
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "thread-1"}}
    
    try:
        resultado = agent.invoke(inputs, config)
        # Retorna o conteúdo da última mensagem (a resposta do agente)
        return resultado["messages"][-1].content
    except Exception as e:
        return f"Erro na execução do agente: {str(e)}"