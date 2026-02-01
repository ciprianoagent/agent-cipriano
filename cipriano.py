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
    search = TavilySearchResults(max_results=3) # Key já injetada no ambiente
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
    
    # Usando gemini-1.5-flash para maior estabilidade no Render
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,
        api_key=api_key
    )
    
    agent = create_react_agent(model=model, tools=tools, state_modifier=system_message)
    
    # Executa e pega a última mensagem da resposta
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "thread-1"}}
    
    resultado = agent.invoke(inputs, config)
    return resultado["messages"][-1].content