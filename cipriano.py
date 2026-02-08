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
# CONFIGURAÇÕES
# ======================================================
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct" 

# ======================================================
# CONFIGURAÇÃO DE LOGS (Resolve o erro: "logger" is not defined)
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BastionEngine")

# ======================================================
# FERRAMENTAS (TOOLS)
# ======================================================
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_web(query: str) -> str:
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web, get_current_datetime]

# ======================================================
# SYSTEM PROMPT (OMITIDO AQUI POR BREVIDADE, MANTENHA O SEU)
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
</output_format>"""

# ======================================================
# INICIALIZAÇÃO (Resolve o erro: "memory" is not defined)
# ======================================================
memory = MemorySaver() # Objeto de persistência definido globalmente
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY não encontrada no .env")

        model = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0, 
            api_key=groq_key
        )
        
        # AQUI ESTÁ A CORREÇÃO SÊNIOR: state_modifier gerencia o System Prompt
        _agent_instance = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=memory,
            state_modifier=system_prompt_content
        )
    return _agent_instance

# ======================================================
# EXECUÇÃO
# ======================================================
def executar_agente(mensagem_usuario: str, imagem_b64: str = None, session_id: str = "default_session"):
    try:
        agent = get_agent()
        
        # 1. Monta o payload (Texto + Imagem Opcional)
        content_payload = [{"type": "text", "text": mensagem_usuario}]
        
        if imagem_b64:
            # Garante o prefixo data:image para o modelo vision
            img_url = imagem_b64 if imagem_b64.startswith("data:") else f"data:image/jpeg;base64,{imagem_b64}"
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
            
        user_message = HumanMessage(content=content_payload)
        
        # 2. Configuração da thread de memória
        config = {"configurable": {"thread_id": session_id}}
        
        # 3. Invoca o agente (O System Prompt já está no state_modifier)
        resultado = agent.invoke({"messages": [user_message]}, config)

        return resultado["messages"][-1].content

    except Exception as e:
        logger.error(f"Erro no Agente: {e}", exc_info=True)
        return "⚠️ Ocorreu um erro interno no processamento do Bastion."