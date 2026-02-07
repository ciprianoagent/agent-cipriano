import os
import datetime
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================================================
# CONFIGURAÇÕES E CONSTANTES
# ======================================================
# Usamos o Llama 3.2 11B Vision (ou 90B) para processar imagens + texto
MODEL_ID = "llama-3.2-11b-vision-preview" 

# ======================================================
# FERRAMENTAS (TOOLS)
# ======================================================

@tool
def get_current_datetime() -> str:
    """
    Retorna a data e hora atual. 
    Essencial para verificar se o suporte está em horário comercial ou feriados,
    ou para validar logs de transações recentes.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_web(query: str) -> str:
    """
    Ferramenta de busca na Web (Tavily).
    
    Use APENAS para:
    1. Pesquisar códigos de erro DESCONHECIDOS que não estão na base interna.
    2. Verificar status atual de serviços externos (AWS, Redes de Adquirência).
    3. Buscar documentações técnicas recentes de bandeiras (Visa/Mastercard) pós-2024.
    
    NÃO USE para:
    - Perguntas sobre a GSurf (Telefones, Endereços, Pix). Use seu conhecimento interno.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada no ambiente."
    
    try:
        search = TavilySearchResults(max_results=2) 
        return search.invoke(query)
    except Exception as e:
        return f"Falha na busca web: {str(e)}"

# Lista de ferramentas disponíveis para o agente
tools = [search_web, get_current_datetime]

# ======================================================
# SYSTEM PROMPT (CÉREBRO DO CIPRIANO)
# ======================================================
system_prompt_content = """
<persona>
Você é o **Cipriano**, Engenheiro de Soluções Sênior e Especialista em Meios de Pagamento da GSurf.
Sua inteligência é técnica, precisa e orientada a solução. Você não "acha", você "diagnostica".
</persona>

<diretrizes>
1. **Postura:** Resolutiva e Técnica. Não peça desculpas excessivas. Vá direto ao ponto.
2. **Imagens:** Se o usuário enviar uma imagem (print de erro, foto de maquininha), ANALISE a imagem visualmente antes de responder. Extraia códigos de erro, textos da tela e luzes acesas.
3. **Foco:** Diagnosticar a falha na cadeia (Emissor -> Bandeira -> Adquirente -> GSurf/TEF).
</diretrizes>

<base_de_conhecimento>
**CANAIS CRÍTICOS:**
* Suporte Técnico 24/7: **0800-644-4833**
* Telefone Geral: (48) 3254-8900
* Comercial: (48) 3254-8700 | comercial@gsurfnet.com
* Site: www.gsurfnet.com

**LÓGICA DE DIAGNÓSTICO (Chain of Thought):**
* Erro "Saldo Insuficiente/Negada": Culpa do **Emissor** (Banco do cliente).
* Erro "Falha de Comunicação": Verificar Internet Local, VPN ou Instabilidade na **Adquirente**.
* Erro "Cartão Inválido": Problema no Chip ou **Bandeira**.
* Erro no E-commerce (CNP): Validar integração API, CVV e ferramentas antifraude (PCI-DSS obrigatório).

**TABELA PIX (SiTef):**
* Itaú/Bradesco: Exigem apenas **Chave Pix**.
* BB/Santander/Cielo/MercadoPago: Exigem **Client ID** + **Client Secret** + **Chave Pix**.
* Sicoob/Sicredi: Exigem **CNPJ Conta** + **Client ID** + **Client Secret** + **Chave Pix**.

**INTEGRAÇÃO ANDROID (M-SITEF):**
Action: `br.com.softwareexpress.sitef.msitef`
Params: `empresaSitef`, `modalidade` (110=Crédito, 111=Débito), `transacoesHabilitadas`.
</base_de_conhecimento>
"""

# ======================================================
# INICIALIZAÇÃO DA MEMÓRIA
# ======================================================
# Inicializa a memória para persistir conversas entre chamadas
memory = MemorySaver()

# Variável global para armazenar o agente (evita recriar a cada chamada)
_agent_instance = None

def get_agent():
    """Singleton para instanciar o agente apenas uma vez."""
    global _agent_instance
    if _agent_instance is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Erro CRÍTICO: GROQ_API_KEY não configurada.")

        model = ChatGroq(
            model=MODEL_ID,
            temperature=0.0, # Precisão máxima
            api_key=groq_key,
            max_retries=2
        )
        
        # Cria o agente com memória
        _agent_instance = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=memory,
            state_modifier=system_prompt_content # Novo padrão do LangGraph para System Prompt
        )
    return _agent_instance

# ======================================================
# FUNÇÃO DE EXECUÇÃO
# ======================================================
def executar_agente(mensagem_usuario: str, imagem_b64: str = None, session_id: str = "default_session"):
    """
    Executa o agente Cipriano com suporte a visão e memória.
    
    Args:
        mensagem_usuario (str): Texto do usuário.
        imagem_b64 (str): String base64 da imagem (sem o prefixo 'data:image...').
        session_id (str): ID único da sessão para manter histórico da conversa.
    """
    try:
        agent = get_agent()
        
        # Construção da mensagem (Multimodal ou Texto Simples)
        content_payload = []
        
        # 1. Adiciona o texto
        content_payload.append({"type": "text", "text": mensagem_usuario})
        
        # 2. Adiciona a imagem se existir (Formato compatível com OpenAI/Groq Vision)
        if imagem_b64:
            # Garante que o cabeçalho data URI esteja correto
            if not imagem_b64.startswith("data:"):
                img_url = f"data:image/jpeg;base64,{imagem_b64}"
            else:
                img_url = imagem_b64
                
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
            
        user_message = HumanMessage(content=content_payload)

        # Configuração da Sessão (Thread)
        config = {"configurable": {"thread_id": session_id}}

        # Invoca o agente
        # stream_mode="values" retorna todas as mensagens, pegamos a última
        resultado = agent.invoke({"messages": [user_message]}, config)

        ultima_mensagem = resultado["messages"][-1]
        
        return ultima_mensagem.content

    except Exception as e:
        # Log de erro real (em produção usar logger)
        print(f"ERRO NO AGENTE: {e}")
        return f"Sistema GSurf informa: Ocorreu uma instabilidade no processamento da sua solicitação. Detalhe técnico: {str(e)}"

# ======================================================
# EXEMPLO DE USO (Teste local)
# ======================================================
if __name__ == "__main__":
    # Teste 1: Pergunta Simples
    print("--- Teste 1: Texto ---")
    resp = executar_agente("Quais os telefones do suporte?", session_id="user_123")
    print(resp)
    
    # Teste 2: Memória (Contexto)
    print("\n--- Teste 2: Memória ---")
    resp = executar_agente("E qual o horário de atendimento desse número?", session_id="user_123")
    print(resp)