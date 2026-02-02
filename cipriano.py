import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================================================
# TOOL — Web Search (OSINT & Tech Docs)
# ======================================================
@tool
def search_web(query: str) -> str:
    """
    Busca informações técnicas atualizadas, documentações de APIs,
    manuais de SiTEF/POS e conhecimentos gerais na internet.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    # Reduzido para evitar estouro de tokens
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# PROMPT ENGENHEIRO DE PAGAMENTOS
# ======================================================
system_prompt_content = """
Você é o Agente Especialista da GSurf (codinome Cipriano).
Sua missão é ser uma autoridade absoluta em TI e Meios de Pagamentos.
Você agora possui VISÃO COMPUTACIONAL: Se o usuário enviar uma imagem, analise-a com cuidado.

### DIRETRIZES:
1. **Identidade:** Profissional, técnico, preciso, mas acessível.
2. **Visão:** Se receber uma imagem de erro, leia o código e o texto da tela. Se for um diagrama, explique o fluxo.
3. **Especialista em TI/Pagamentos:**
    - Ecossistema: Portador -> POS -> Gateway (SiTEF) -> Adquirente -> Bandeira -> Emissor.
    - Conectividade: VPN IPsec, MPLS,
    - Conceitos: BIN, ISO 8583, Adquirência.
4. **gsurfnet.com:** Tudo sobre a empresa, o que ela faz, o que ela entrega.     

### ROTEAMENTO MENTAL:
- Se for imagem de erro: Identifique o código e sugira solução.
- Se for pergunta técnica: Use seu conhecimento ou busque na web se necessário.
"""

def executar_agente(mensagem_usuario: str, imagem_b64: str = None):
    """
    Executa o agente. Se imagem_b64 for fornecida, envia para o modelo Vision.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "Erro CRÍTICO: GROQ_API_KEY não configurada."

    # --- MUDANÇA IMPORTANTE: MODELO VISION ---
    # Usamos o Llama 3.1 Vision para poder enxergar as imagens
    model = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.4,
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        # Montagem da mensagem do usuário
        if imagem_b64:
            # Se tem imagem, montamos o payload multimodal
            content_payload = [
                {"type": "text", "text": mensagem_usuario or "Analise esta imagem."},
                {
                    "type": "image_url", 
                    "image_url": {"url": imagem_b64} # O frontend já manda com "data:image/..."
                }
            ]
            user_message = HumanMessage(content=content_payload)
        else:
            # Se é só texto
            user_message = HumanMessage(content=mensagem_usuario)

        inputs = {
            "messages": [
                ("system", system_prompt_content),
                user_message
            ]
        }

        config = {"configurable": {"thread_id": "session-1"}}

        resultado = agent.invoke(inputs, config)

        ultima_mensagem = resultado["messages"][-1]
        if hasattr(ultima_mensagem, "content"):
            return ultima_mensagem.content

        return str(ultima_mensagem)

    except Exception as e:
        return f"Sistema GSurf informa: Erro no processamento. ({str(e)})"