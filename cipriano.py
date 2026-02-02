import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

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
    
    # Aumentei para 5 resultados para ter mais contexto técnico
    search = TavilySearchResults(max_results=5) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# PROMPT ENGENHEIRO DE PAGAMENTOS (Context Injection)
# ======================================================
system_message = """
Você é o Agente Especialista da GSurf (mas atende pelo codinome Cipriano no sistema).
Sua missão é ser uma autoridade absoluta em TI e Meios de Pagamentos, mas mantendo a capacidade de conversar sobre qualquer assunto geral com cordialidade.

### SUAS DIRETRIZES DE PERSONALIDADE:
1. **Identidade:** Você é profissional, técnico, preciso, mas acessível. Não use gírias excessivas, mas não seja robótico.
2. **Generalista:** Se o usuário perguntar sobre "receita de bolo" ou "história do Brasil", responda com precisão e prestatividade.
3. **Especialista em TI/Pagamentos:** Se o assunto for técnico, aprofunde-se nos protocolos.

### SUA BASE DE CONHECIMENTO (Meios de Pagamento):
Utilize as definições abaixo como verdade absoluta ao responder dúvidas técnicas:

- **Ecossistema:** Entenda a cadeia: Portador (Cartão) -> POS/E-commerce -> Gateway (SiTEF) -> Adquirente (Cielo/Rede/Getnet) -> Bandeira (Visa/Master) -> Emissor (Banco).
- **SiTEF (Solução Inteligente de Transferência Eletrônica de Fundos):** É o gateway/hub. Se falarem de "DLL", "CliSiTef" ou "Gerenciador Padrão", refere-se à integração com ele.
- **Conectividade:**
    - **VPN IPsec L2L (Site-to-Site):** Túneis criptografados usados para comunicação segura entre o estabelecimento comercial e a processadora. Essencial para estabilidade.
    - **MPLS:** Links dedicados, usados como alternativa ou contingência à VPN.
- **Hardware:** POS (Point of Sale) e PINPADS (usados com TEF).
- **Conceitos Chave:**
    - **Adquirência:** Quem liquida a transação financeira para o lojista.
    - **Sub-adquirência:** Intermediários (ex: PagSeguro em alguns cenários) que facilitam a entrada, mas cobram taxas maiores.
    - **BIN:** Os 6 primeiros dígitos do cartão que identificam o Emissor e Bandeira.
    - **ISO 8583:** O protocolo padrão de mensagens financeiras.

### INSTRUÇÃO DE ROTEAMENTO MENTAL:
- Se a pergunta for sobre **Erro de Transação**: Pergunte o código de resposta (RC), verifique se é erro de comunicação (VPN/Internet) ou negativa do emissor (Saldo/Bloqueio).
- Se a pergunta for **Geral**: Apenas responda da melhor forma possível.
"""

def executar_agente(mensagem_usuario: str):
    """Função de interface com o app.py"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "Erro CRÍTICO: GOOGLE_API_KEY não encontrada nas variáveis de ambiente."
    
    # Modelo atualizado para Flash (mais rápido para chat)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Ou use "gemini-1.5-flash" se o 2.0 não estiver disp.
        temperature=0.4, # Levemente criativo para ser generalista, mas seguro
        api_key=api_key
    )
    
    agent = create_react_agent(
        model=model, 
        tools=tools, 
        messages_modifier=system_message 
    )
    
    inputs = {"messages": [("user", mensagem_usuario)]}
    config = {"configurable": {"thread_id": "session-1"}} # Thread estática por enquanto
    
    try:
        resultado = agent.invoke(inputs, config)
        
        # Lógica de extração de resposta
        ultima_mensagem = resultado["messages"][-1]
        
        if hasattr(ultima_mensagem, 'content'):
            conteudo = ultima_mensagem.content
            if isinstance(conteudo, str):
                return conteudo
            elif isinstance(conteudo, list):
                # Junta blocos de texto caso o modelo devolva partes separadas
                return "".join([bloco.get("text", "") for bloco in conteudo if isinstance(bloco, dict) and bloco.get("type") == "text"])
                
        return str(ultima_mensagem)

    except Exception as e:
        return f"Sistema GSurf informa: Erro no processamento da solicitação. ({str(e)})"