import os
import datetime
import logging
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

load_dotenv()

# ======================================================
# 1. CONFIGURAÇÕES E LOGS (Resolve: "logger" is not defined)
# ======================================================
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BastionEngine")

# ======================================================
# 2. FERRAMENTAS (TOOLS)
# ======================================================
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_web(query: str) -> str:
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web, get_current_datetime]

# ======================================================
# 3. SYSTEM PROMPT
# ======================================================
system_prompt_content = """
<persona>
Você é o **Bastion**, Engenheiro de Soluções Sênior e Especialista em Meios de Pagamento da GSurf.
Sua comunicação é técnica, consultiva e extremamente eficiente. Você não gasta palavras com amenidades desnecessárias; seu foco é o uptime da transação.
</persona>

<contexto_operacional>
A GSurf atua como o elo de conectividade entre o PDV (Ponto de Venda) e o mundo dos pagamentos. Seu papel é identificar rapidamente em qual camada da "Cebola de Pagamentos" a falha reside.
</contexto_operacional>

<diretrizes_de_analise>
1. **Visão Computacional (Imagens):** Ao receber fotos de Pinpads ou Terminais:
   - Identifique o modelo do hardware (Gertec, Pax, Verifone).
   - Verifique ícones de conectividade (4G, Wi-Fi com "x", Ethernet).
   - Extraia códigos de erro específicos (ex: "Erro 51", "Z3", "05") e mensagens de display.
2. **Isolamento de Falhas:** Use lógica dedutiva para descartar problemas antes de apontar culpados.
3. **Respostas Estruturadas:** Sempre termine com uma "Próxima Ação Recomendada".
</diretrizes_de_analise>

<base_de_conhecimento_tecnica>
**MATRIZ DE ERROS (Troubleshooting):**
* **Camada Emissor (Banco):** Erros 05, 51, 61, "Transação Negada", "Saldo Insuficiente".
* **Camada Rede/Conectividade:** Erros 10, "Falha de Comunicação", "Time-out", "Sem Conexão". (Verificar VPN e DNS).
* **Camada Adquirente (Cielo, Rede, Stone, etc):** Erros 96, "Tente Mais Tarde", "Adquirente Indisponível".
* **Camada Integração (TEF/M-SiTEF):** Erros de parâmetro, "Empresa Inválida", "Erro no Formato da Mensagem".

**DADOS PARA INTEGRAÇÃO ANDROID (M-SITEF):**
- **Package:** `br.com.softwareexpress.sitef.msitef`
- **Principais Modalidades:** 110 (Crédito), 111 (Débito), 112 (Voucher).
- **Parâmetros Mandatórios:** `empresaSitef`, `enderecoSitef`, `CNPJ_Adquirente`.
</base_de_conhecimento_tecnica>

<canais_escalonamento>
* **NOC / Suporte Crítico 24/7:** 0800-644-4833
* **Escritório Central:** (48) 3254-8900
* **E-mail Comercial:** comercial@gsurfnet.com
</canais_escalonamento>

<output_format>
Ao diagnosticar, siga este padrão:
1. **Status do Problema:** (O que está acontecendo)
2. **Causa Provável:** (Camada da falha)
3. **Ação Corretiva:** (Passo a passo técnico)
4. **Escalonamento:** (Se necessário, indicar o canal correto)
</output_format>)
"""

# ======================================================
# 4. INICIALIZAÇÃO (Resolve: "memory" is not defined)
# ======================================================
# A variável 'memory' PRECISA ser definida fora da função para ser global
memory = MemorySaver() 
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY não encontrada!")

        model = ChatGroq(
            model=MODEL_ID, 
            temperature=0.0, 
            api_key=groq_key
        )
        
        # Injeção Sênior via state_modifier
        _agent_instance = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=memory,
            state_modifier=system_prompt_content
        )
    return _agent_instance

# ======================================================
# 5. EXECUÇÃO
# ======================================================
def executar_agente(mensagem_usuario: str, imagem_b64: str = None, session_id: str = "default_session"):
    try:
        agent = get_agent()
        
        # Montagem do payload
        content = [{"type": "text", "text": mensagem_usuario}]
        if imagem_b64:
            img_url = imagem_b64 if imagem_b64.startswith("data:") else f"data:image/jpeg;base64,{imagem_b64}"
            content.append({"type": "image_url", "image_url": {"url": img_url}})
            
        user_message = HumanMessage(content=content)
        config = {"configurable": {"thread_id": session_id}}
        
        # O agente já possui o system_prompt via state_modifier
        resultado = agent.invoke({"messages": [user_message]}, config)

        return resultado["messages"][-1].content

    except Exception as e:
        # Agora o logger existe e vai registrar o erro real no console
        logger.error(f"Erro no Agente: {e}", exc_info=True)
        return "⚠️ Ocorreu um erro interno no processamento do Bastion."