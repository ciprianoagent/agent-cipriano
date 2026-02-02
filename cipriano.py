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
    Executa o agente. Versão ajustada para Llama 3.1 (Apenas Texto).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "Erro CRÍTICO: GROQ_API_KEY não configurada."

    # Voltamos para o modelo rápido de texto
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

        # --- CORREÇÃO DO ERRO 400 ---
        # O Llama 3.1-8b NÃO aceita objetos de imagem, apenas strings.
        # Então, mesmo que venha uma imagem do front, nós preparamos apenas o texto.
        
        texto_final = mensagem_usuario

        if imagem_b64:
            # Adicionamos uma nota interna para o agente saber que houve uma tentativa de envio
            texto_final += "\n\n[Sistema: O usuário anexou uma imagem, mas seu modelo atual não possui visão computacional. Avise o usuário que você não consegue ver a imagem e peça para ele descrever o que está nela.]"

        # Enviamos como string simples (Isso resolve o erro 'content must be a string')
        user_message = HumanMessage(content=texto_final)

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