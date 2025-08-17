# Agentic RAG System

A powerful Retrieval-Augmented Generation (RAG) system that combines document knowledge with web search capabilities using Ollama and LangChain. This system can answer queries by retrieving information from both indexed documents and real-time web searches.

## Features

- **Document Retrieval**: Index and search through PDF and text documents using FAISS vector store
- **Web Search Integration**: Real-time web search using DuckDuckGo
- **Intelligent Agent**: Uses ReAct framework for step-by-step reasoning and tool selection
- **Ollama Integration**: Leverages local LLM models for privacy and performance
- **Flexible Configuration**: Environment-based configuration for easy deployment

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Required Ollama models:
  - `llama3.2` (or your preferred LLM model)
  - `mxbai-embed-large:latest` (embedding model)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ekluvtech/agenticrag.git
   cd agenticrag
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai/)
   
   # Pull required models
   ollama pull llama3.2
   ollama pull mxbai-embed-large:latest
   ```

## Usage

### 1. Building the Knowledge Base

First, build your document knowledge base:

```bash
python buildKnowledgeBase.py
```

This script will:
- Load documents from the `FOLDER_PATH` directory
- Process PDF and text files
- Create embeddings using the specified embedding model
- Build and save a FAISS vector index

**Supported file formats:**
- PDF files (`.pdf`)
- Text files (`.txt`)

### 2. Running the Agentic RAG System

```bash
python main.py
```

The system will:
- Load the pre-built knowledge base
- Initialize the agent with document retrieval and web search tools
- Process queries using intelligent tool selection
- Provide answers with source attribution

## Configuration

### Config params

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `LLM_MODEL` | LLM model name | `llama3.2` |
| `EMBED_MODEL` | Embedding model name | `mxbai-embed-large:latest` |
| `FAISS_INDEX_NAME` | FAISS index name | `faiss_idx` |
| `FOLDER_PATH` | Path to documents folder | `<<PATH_TO_DIRECOTRY>>/faissdata/data` |
| `INDEX_STORAGE_PATH` | Path to store FAISS index | `<<PATH_TO_DIRECOTRY>>/faissdata/index` |

### Model Requirements

The system requires two Ollama models:

1. **LLM Model**: For text generation and reasoning (e.g., `llama3.2`)
2. **Embedding Model**: For document vectorization (e.g., `mxbai-embed-large:latest`)

## Project Structure

```
agenticrag/
├── main.py                 # Main agentic RAG system
├── buildKnowledgeBase.py   # Document indexing and knowledge base builder
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── owasp-top-10.pdf      # Example document
```

## How It Works

1. **Document Processing**: Documents are loaded, split into chunks, and embedded using the specified embedding model
2. **Vector Storage**: FAISS vector store is created and saved locally for fast retrieval
3. **Agent Initialization**: A ReAct agent is created with two tools:
   - **DocumentRetriever**: Searches the indexed knowledge base
   - **DuckDuckGoSearch**: Performs real-time web searches
4. **Query Processing**: The agent intelligently selects tools based on the query type
5. **Response Generation**: Combines information from multiple sources to provide comprehensive answers

## Example Usage

```python
# The system automatically handles queries like:
query = "What are the top 10 OWASP vulnerabilities?"
# This will use both document retrieval and web search to provide a comprehensive answer
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check the `OLLAMA_URL` configuration

2. **Model Not Found**
   - Pull required models: `ollama pull llama3.2`
   - Verify model names in configuration

3. **Document Loading Errors**
   - Check file permissions and paths
   - Ensure documents are in supported formats (PDF, TXT)

4. **FAISS Index Issues**
   - Delete existing index and rebuild: `rm -rf /path/to/index/*`
   - Run `buildKnowledgeBase.py` again

### Performance Tips

- Use SSD storage for better FAISS performance
- Adjust chunk size in `buildKnowledgeBase.py` for optimal retrieval
- Consider using GPU-enabled models for faster processing

## Dependencies

- `langchain`: Core RAG framework
- `langchain-community`: Community integrations
- `langchain_ollama`: Ollama integration
- `langchain_core`: Core LangChain components
- `langgraph`: Agent orchestration
- `duckduckgo-search`: Web search functionality
- `faiss-cpu`: Vector similarity search
- `pypdf`: PDF processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration
3. Open an issue on GitHub with detailed error information 
