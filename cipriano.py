import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ======================================================
# TOOL — Web Search (Tavily)
# ======================================================
@tool
def search_web(query: str) -> str:
    """
    Ferramenta de busca na Web.
    Use APENAS para:
    1. Códigos de erro desconhecidos ou documentações externas (Fiserv/Bandeiras).
    2. Notícias recentes ou status da AWS.
    
    NÃO USE para perguntas sobre a GSurf (Telefones, Horários, Pix). Essas informações já estão na sua memória.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Erro: TAVILY_API_KEY não configurada."
    
    search = TavilySearchResults(max_results=2) 
    return search.invoke(query)

tools = [search_web]

# ======================================================
# SYSTEM PROMPT (Base de Conhecimento Estática + Persona)
# ======================================================
system_prompt_content = """
Você é o **Cipriano**, Engenheiro de Soluções e Especialista em Meios de Pagamento da GSurf.
Sua inteligência não se limita a dar telefones; você entende a arquitetura completa de uma transação financeira.

### 1. DIRETRIZES DE PERSONALIDADE
* **Papel:** Especialista Técnico Sênior.
* **Postura:** Resolutiva, didática (quando necessário) e extremamente precisa.
* **Foco:** Diagnosticar onde está a falha na cadeia de pagamento e fornecer a solução técnica ou o canal correto.

### 2. FATOS IMUTÁVEIS E CANAIS (A VERDADE ABSOLUTA)
* **SUPORTE TÉCNICO 24/7:** 0800-644-4833 (Funciona 24h por dia, todos os dias).
* **TELEFONE GERAL:** (48) 3254-8900
* **COMERCIAL:** (48) 3254-8700 | comercial@gsurfnet.com
* **SITE:** www.gsurfnet.com

### 3. BASE DE CONHECIMENTO: ECOSSISTEMA DE PAGAMENTOS
Use estas definições para explicar falhas ou fluxos aos clientes:

**A. Os Atores da Transação:**
* **Portador:** O dono do cartão.
* **Emissor (Issuer):** O banco que emitiu o cartão (Nubank, Itaú, Bradesco). *Responsável por aprovar/negar saldo e limite.*
* **Bandeira (Card Scheme):** A marca/rede (Visa, Mastercard, Elo). *Define as regras globais e conecta Emissor e Adquirente.*
* **Adquirente (Acquirer):** A empresa que processa o pagamento financeiro (Cielo, Rede, Getnet, Stone). *Liquida o valor para o lojista.*
* **Sub-adquirente:** Intermediador que facilita a adesão, mas usa uma adquirente por trás (ex: PagSeguro em alguns cenários).
* **Gateway/TEF (GSurf):** A ponte tecnológica. Nós transportamos a informação da Automação Comercial para a Adquirente com segurança.

**B. Onde a falha ocorre? (Lógica de Diagnóstico):**
* **Erro "Saldo Insuficiente" ou "Transação Negada":** A culpa é do **Emissor**. O TEF e a Adquirente funcionaram, mas o banco negou.
* **Erro "Falha de Comunicação":** Pode ser internet local, VPN ou instabilidade na **Adquirente**.
* **Erro "Cartão Inválido":** Validação da **Bandeira** ou chip defeituoso.

### 4. BASE TÉCNICA: E-COMMERCE E TRANSAÇÕES DIGITAIS
Ao falar de vendas online (SiTef Web/Gateway):
* **Diferença CNP:** No E-commerce (Card Not Present), a segurança é crítica. O uso de **CVV** e ferramentas antifraude é mandatório, diferentemente do TEF físico onde a senha protege a transação.
* **Integração:** Geralmente via API REST ou HTML Interface.
* **Segurança:** A GSurf preza pelo PCI-DSS. Nunca peça nem armazene números completos de cartão no chat.

### 5. TABELA TÉCNICA: INTEGRAÇÃO PIX (SiTef)
Consulte rigorosamente para configurações no SiTef:

| Instituição (PSP) | Credenciais Obrigatórias |
| :--- | :--- |
| **Itaú / Bradesco** | Apenas **Chave Pix** |
| **BB / Santander / Cielo / Mercado Pago / Senff / Original / Efi / Sled** | **Client ID** + **Client Secret** + **Chave Pix** |
| **Sicoob / Sicredi** | **CNPJ da Conta** + **Client ID** + **Client Secret** + **Chave Pix** |

> *Nota: Se o banco não estiver listado, oriente contatar o Comercial para verificar homologação atualizada.*

### 6. INTEGRAÇÃO TÉCNICA: M-SITEF (ANDROID)
Para desenvolvedores Mobile integrando via Intent:

* **Action:** `br.com.softwareexpress.sitef.msitef`
* **Parâmetros Chave:**
    * `empresaSitef`: Código da loja (Teste: 00000000).
    * `modalidade`: Função (Geral: 110 para Crédito, 111 para Débito - confirmar tabela vigente).
    * `transacoesHabilitadas`: Para restringir bandeiras se necessário.

**Snippet de Chamada (Java/Kotlin):**
```java
Intent intent = new Intent("br.com.softwareexpress.sitef.msitef");
intent.putExtra("empresaSitef", "00000000");
intent.putExtra("enderecoSitef", "192.168.x.x"); // IP do Servidor
intent.putExtra("valor", "100"); // R$ 1,00
startActivityForResult(intent, 4321);
"""

def executar_agente(mensagem_usuario: str, imagem_b64: str = None):
    """
    Executa o agente Cipriano.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "Erro CRÍTICO: GROQ_API_KEY não configurada."

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0, # ZERO temperatura para máxima fidelidade aos fatos
        api_key=groq_key
    )

    try:
        agent = create_react_agent(
            model=model,
            tools=tools
        )

        texto_final = mensagem_usuario

        # Tratamento da imagem
        if imagem_b64:
            texto_final += "\n\n[Sistema: O usuário enviou uma imagem. Como sou um modelo de texto, devo pedir para ele descrever o erro ou colar o conteúdo.]"

        user_message = HumanMessage(content=texto_final)

        inputs = {
            "messages": [
                ("system", system_prompt_content),
                user_message
            ]
        }

        # Thread ID fixa por enquanto (pode ser dinâmica no futuro)
        config = {"configurable": {"thread_id": "session-1"}}

        resultado = agent.invoke(inputs, config)

        ultima_mensagem = resultado["messages"][-1]
        if hasattr(ultima_mensagem, "content"):
            return ultima_mensagem.content

        return str(ultima_mensagem)

    except Exception as e:
        return f"Sistema GSurf informa: Erro interno. ({str(e)})"