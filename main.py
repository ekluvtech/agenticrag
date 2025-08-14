import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun  # Import DuckDuckGo search tool
from config import *
from langchain_ollama import ChatOllama
import json
from langchain_core.messages import AIMessage, ToolMessage
from ddgs import DDGS

# Load and process document
# (Assuming document loading and splitting is handled elsewhere as per your original code)

# Create vector store
embed_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore = FAISS.load_local(f"{INDEX_STORAGE_PATH}", embed_model, f"{FAISS_INDEX_NAME}", allow_dangerous_deserialization=True)

# Define retrieval tool
retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "score_threshold": 0.9})

# Initialize DuckDuckGo search tool
ddg = DDGS()
# Define tools list with both DocumentRetriever and DuckDuckGo search
tools = [
    
    Tool(
        name="DuckDuckGoSearch",
        func=ddg.text,
        description="Search the web using DuckDuckGo to find relevant information for the query.",
    ),
    Tool(
        name="DocumentRetriever",
        func=retriever.get_relevant_documents,
        description="Retrieve relevant document chunks based on a query.",
    )
]

query = "what are the top 10 OWASP vulnerabilities?"
llm = ChatOllama(model=LLM_MODEL, format="json", base_url=OLLAMA_URL)
prompt_template = PromptTemplate(
    input_variables=["messages"],
    template="""You are an advanced AI assistant with access to the following tools to help answer user queries accurately and efficiently:

1. **DuckDuckGoSearch**: A tool that allows you to search the web using DuckDuckGo to find relevant information for the query. Use this tool when you need up-to-date or external information not available in your internal knowledge base.
2. **DocumentRetriever**: A tool that retrieves relevant document chunks based on a query. Use this tool when the query requires specific information from a predefined set of documents or a knowledge base.

**Instructions:**
- Analyze the user's query, which is contained within the conversation history provided in the `messages` input. The `messages` list includes the user's latest query and may include prior messages, such as `AIMessage` (AI responses) or `ToolMessage` (tool outputs).
- Identify the latest user query from the `messages` list to determine which tool(s) are most appropriate to use.
- If the query requires real-time or external information (e.g., recent events, news, or web-based data), use **DuckDuckGoSearch** to fetch relevant results.
- If the query asks for specific details that might be contained in a predefined knowledge base or document set (e.g., technical details, internal data, or specific references), use **DocumentRetriever** to fetch relevant document chunks.
- If both tools could be useful, prioritize **DocumentRetriever** for precise, document-based answers, and use **DuckDuckGoSearch** to supplement with additional context or recent information if needed.
- Always provide a clear, concise, and accurate response based on the information retrieved. If the tools do not provide sufficient information, state this clearly and provide the best possible answer based on your internal knowledge.
- Format your response in a structured and readable way, citing the source of information (e.g., web search or document retrieval) when applicable.
- If no tools are needed because the query can be answered directly with your internal knowledge, do so without invoking the tools.
- Use the ReAct framework to reason step-by-step, documenting your thought process in the agent_scratchpad. The scratchpad should include your reasoning, tool selection, and intermediate steps.
- If the `messages` list contains `ToolMessage` entries, incorporate their content (e.g., tool outputs) into your reasoning and response as appropriate.

**Conversation History:**  
{messages}

**Response Format:**  
- **Answer**: [Provide the answer to the query in a clear and concise manner.]  
- **Source**: [Indicate whether the information came from DuckDuckGoSearch, DocumentRetriever, or internal knowledge.]  
- **Additional Notes** (if applicable): [Include any relevant context, limitations, or clarifications.]
"""
)
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt_template)

# Custom action for agentic behavior
def custom_action(query_result):
    #pretty_json = json.dumps(query_result, indent=2)
    #print(query_result)
    if isinstance(query_result, dict) and "messages" in query_result:
        # Extract the last AIMessage from the messages list
        messages = query_result["messages"]
        for message in reversed(messages):  # Iterate in reverse to get the latest AIMessage
            if isinstance(message, AIMessage) and message.content:
                try:
                    # Parse the content as JSON
                    content_json = json.loads(message.content)
                    # Extract the 'output' field if it exists
                    # print(message.content)
                    return content_json.get("output", str(message.content))
                except json.JSONDecodeError:
                    # If content is not JSON, return it as is
                    return message.content
        # If no valid AIMessage is found, return the raw result as a string
        return str(query_result)
    return str(query_result)

# Run query
response = custom_action(agent.invoke({"messages": [{"role": "user", "content": "list out top 10 OWASP vulnerabilities"}]}))

# Print or process the output
print(response)