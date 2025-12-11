# Agent System - Document Search

A single agentic system using LangChain with RAG and Internet Search capabilities. This system can search insurance documents by contract ID and perform web searches.

## Features

- **RAG Tool**: Searches insurance documents stored in a vector database by contract ID (e.g., HD-7961)
- **Internet Search Tool**: Performs web searches using Google Serper API
- **LangChain Agent**: Uses LangChain's ReAct agent framework to intelligently route queries to the appropriate tool
- **Web Interface**: Gradio-based web UI for easy interaction

## Prerequisites

1. **API Keys**: You need the following API keys in your `.env` file:
   - `OPENAI_API_KEY`: For LLM and embeddings
   - `SERPER_API_KEY`: For internet search (get it from [Serper.dev](https://serper.dev))

2. **Dependencies**: The following packages should be installed:
   ```bash
   # Core dependencies (should already be in project)
   - langchain
   - langchain-community
   - langchain-openai
   - pypdf
   - chromadb
   - gradio
   ```

   To install ChromaDB:
   ```bash
   uv pip install chromadb
   ```

## File Structure

```
4_langgraph/
├── agent.py              # Core Agent System (Backend)
├── agent_web_app.py      # Web Interface (Gradio UI)
├── README.md             # This file
└── agent_chroma_db/      # Vector Database (created at runtime)
```

## Usage

### Option 1: Web Interface (Recommended)

```bash
cd 4_langgraph
python agent_web_app.py
```

This will:
1. Initialize the agent
2. Load or create the vector database
3. Start Gradio server on `http://localhost:7860`
4. Open browser automatically

**Web Interface Features:**
- **Tab 1: Search by Contract ID** - Enter a contract ID (e.g., HD-7961) to search
- **Tab 2: General Query** - Ask any question about documents or use internet search
- **Tab 3: Admin** - Reload documents if needed

### Option 2: Python API

```python
from agent import Agent

# Initialize the agent with the PDF URL
pdf_url = "https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf"
agent = Agent(pdf_url=pdf_url, vector_db_path="./agent_chroma_db")

# Step 1: Ingest documents (only needed once, or when documents change)
agent.ingest_documents()

# Step 2: Create the agent with tools
agent.create_agent()

# Step 3: Query the agent
# Search for contract ID HD-7961
result = agent.run("What information can you find about contract ID HD-7961?")
print(result)

# Search the internet
result = agent.run("What are the latest trends in insurance certificates?")
print(result)
```

### Option 3: Command Line

```bash
cd 4_langgraph
python agent.py
```

This runs the main() function with example queries.

## How It Works

1. **Document Ingestion**:
   - Downloads the PDF from the specified URL
   - Loads and splits the document into chunks
   - Creates embeddings using OpenAI
   - Stores embeddings in ChromaDB vector database

2. **RAG Tool**:
   - Takes a contract ID as input
   - Searches the vector database for relevant information
   - Returns context-based answers using the LLM

3. **Internet Search Tool**:
   - Uses Google Serper API to search the web
   - Returns current information from the internet

4. **Agent**:
   - Uses LangChain's ReAct (Reasoning + Acting) framework
   - Intelligently decides which tool to use based on the query
   - Can chain multiple tool calls if needed

## Example Queries

### Search by Contract ID:
```python
agent.run("Find information about contract ID HD-7961")
agent.run("What does contract HD-7961 cover?")
```

### Internet Search:
```python
agent.run("What is a certificate of insurance?")
agent.run("Search for recent insurance industry news")
```

### Combined Queries:
```python
agent.run("What information is available about HD-7961, and also search online for similar insurance certificates?")
```

## Configuration

You can customize the agent by modifying the following parameters:

- `vector_db_path`: Path where the vector database is stored (default: `./agent_chroma_db`)
- `model`: LLM model to use (default: `gpt-4o-mini`)
- `chunk_size`: Size of document chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `k`: Number of documents to retrieve (default: 5)

## Notes

- The vector database persists on disk, so you don't need to re-ingest documents unless they change
- The system uses OpenAI embeddings (text-embedding-3-small by default)
- Make sure your `.env` file is in the project root with the required API keys

## Troubleshooting

1. **ChromaDB not found**: Install it with `uv pip install chromadb`
2. **API Key errors**: Ensure `OPENAI_API_KEY` and `SERPER_API_KEY` are in your `.env` file
3. **PDF download fails**: Check your internet connection and the PDF URL
4. **Import errors**: Make sure you're using the correct Python environment (`.venv`)

## Architecture

```
┌─────────────────────────────────┐
│   agent_web_app.py              │  ← Web UI (Gradio)
│   • 3 Tabs                      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   agent.py                      │  ← Backend (LangChain Agent)
│   Agent Class:                  │
│   • RAG Tool                    │
│   • Internet Search Tool        │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────┐
│ ChromaDB │  │  OpenAI  │
│ Vector   │  │  LLM +   │
│ Store    │  │ Embeddings│
└──────────┘  └──────────┘
```

