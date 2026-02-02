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
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "Erro CRÍTICO: GOOGLE_API_KEY não encontrada."
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.4,
        api_key=api_key
    )
    
    # Versão corrigida: O create_react_agent agora costuma aceitar o 
    # state_modifier ou as instruções devem ser injetadas no prompt.
    # Para garantir compatibilidade com as versões 0.2+, usamos state_modifier:
    
    try:
        agent = create_react_agent(
            model=model, 
            tools=tools, 
            # Alterado de messages_modifier para state_modifier
            state_modifier=system_message 
        )
        
        inputs = {"messages": [("user", mensagem_usuario)]}
        config = {"configurable": {"thread_id": "session-1"}}
        
        resultado = agent.invoke(inputs, config)
        
        # O retorno do LangGraph é um dicionário com a chave "messages"
        # Pegamos a última mensagem da lista
        ultima_mensagem = resultado["messages"][-1]
        
        return ultima_mensagem.content

    except Exception as e:
        # Se 'state_modifier' ainda der erro, sua versão é muito antiga.
        # Tente atualizar o pacote: pip install -U langgraph
        return f"Sistema GSurf informa: Erro no processamento. ({str(e)})"