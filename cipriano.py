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
# CONFIGURAÇÕES
# ======================================================
MODEL_ID = "llama-3.2-11b-vision-preview" 

# ======================================================
# FERRAMENTAS (TOOLS)
# ======================================================
@tool
def get_current_datetime() -> str:
    """Retorna a data e hora atual."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_web(query: str) -> str:
    """Busca na Web (Tavily)."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    try:
        search = TavilySearchResults(max_results=2) 
        return search.invoke(query)
    except Exception as e:
        return f"Falha na busca web: {str(e)}"

tools = [search_web, get_current_datetime]

# ======================================================
# SYSTEM PROMPT (O CÉREBRO)
# ======================================================
system_prompt_content = """
<persona>
Você é o **Cipriano**, Engenheiro de Soluções Sênior e Especialista em Meios de Pagamento da GSurf.
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
</output_format>
"""

# ======================================================
# INICIALIZAÇÃO
# ======================================================
memory = MemorySaver()
_agent_instance = None

def get_agent():
    """Cria o agente SEM passar o system prompt na configuração para evitar erros de versão."""
    global _agent_instance
    if _agent_instance is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Erro CRÍTICO: GROQ_API_KEY não configurada.")

        model = ChatGroq(
            model=MODEL_ID,
            temperature=0.0,
            api_key=groq_key,
            max_retries=2
        )
        
        # VERSÃO BLINDADA: Removemos 'state_modifier' e 'messages_modifier'
        # Passaremos o prompt manualmente no invoke.
        _agent_instance = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=memory
        )
    return _agent_instance

# ======================================================
# EXECUÇÃO
# ======================================================
def executar_agente(mensagem_usuario: str, imagem_b64: str = None, session_id: str = "default_session"):
    try:
        agent = get_agent()
        
        # 1. Monta o payload da mensagem do usuário
        content_payload = []
        content_payload.append({"type": "text", "text": mensagem_usuario})
        
        if imagem_b64:
            if not imagem_b64.startswith("data:"):
                img_url = f"data:image/jpeg;base64,{imagem_b64}"
            else:
                img_url = imagem_b64
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
            
        user_message = HumanMessage(content=content_payload)
        
        # 2. Monta a lista de mensagens INJETANDO O SYSTEM PROMPT NO INÍCIO
        # Essa é a forma universal que funciona em qualquer versão do LangGraph
        mensagens_para_enviar = [
            SystemMessage(content=system_prompt_content),
            user_message
        ]

        # 3. Executa com configuração de thread
        config = {"configurable": {"thread_id": session_id}}
        
        resultado = agent.invoke({"messages": mensagens_para_enviar}, config)

        ultima_mensagem = resultado["messages"][-1]
        return ultima_mensagem.content

    except Exception as e:
        print(f"ERRO CRÍTICO NO AGENTE: {e}")
        return f"Sistema GSurf informa: Erro interno de processamento. ({str(e)})"