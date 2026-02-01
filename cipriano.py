import os
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# ======================================================
# ENV — carregar, mas NÃO confiar só nisso
# ======================================================
load_dotenv()

# ======================================================
# LLM FACTORY (lazy)
# ======================================================
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY ou GEMINI_API_KEY não encontrada no ambiente. "
            "No langgraph dev, use: set GOOGLE_API_KEY=..."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
        api_key=api_key
    )

# ======================================================
# TOOL — Web Search (OSINT)
# ======================================================
@tool
def search_web(query: str) -> str:
    """
    Busca informações públicas e atuais na internet (OSINT).
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "TAVILY_API_KEY não configurada."

    search = TavilySearchResults(
        max_results=5,
        api_key=tavily_key
    )
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT — CIPRIANO
# ======================================================
system_message = """
Você é Cipriano, um agente estratégico de Inteligência e Segurança,
com a presença, autoridade e frieza calculada de Don Corleone.

- Seja direto.
- Não invente dados.
- Use busca para fatos atuais.
- Atue apenas em contextos legais e éticos.
"""

# ======================================================
# GRAPH FACTORY (isso é o que o langgraph dev importa)
# ======================================================
def build_graph():
    model = get_llm()

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=system_message
    )

# ======================================================
# ENTRYPOINT DO LANGGRAPH
# ======================================================
graph = build_graph()
