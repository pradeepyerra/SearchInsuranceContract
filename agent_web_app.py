"""
Web Interface for Agent System
Gradio-based web UI for searching insurance documents by contract ID
"""

import os
import sys

# Fix Gradio static files issue
os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")

import gradio as gr
from agent import Agent
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)  # Fallback to current directory

# Initialize the agent (will be loaded when the app starts)
agent_instance = None
pdf_url = "https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf"
vector_db_path = "./agent_chroma_db"


def initialize_agent():
    """Initialize the agent (only once)."""
    global agent_instance
    
    if agent_instance is None:
        try:
            print("Initializing Agent...")
            agent_instance = Agent(pdf_url=pdf_url, vector_db_path=vector_db_path)
            
            # Check if vector store already exists, if not, ingest documents
            if not Path(vector_db_path).exists():
                print("Vector store not found. Ingesting documents...")
                agent_instance.ingest_documents()
            else:
                print("Loading existing vector store...")
                # Load existing vector store and recreate RAG chain
                from langchain_openai import OpenAIEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser
                
                try:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
                    embeddings = OpenAIEmbeddings(api_key=api_key)
                    agent_instance.vector_store = Chroma(
                        persist_directory=vector_db_path,
                        embedding_function=embeddings
                    )
                    
                    # Recreate RAG chain (reduced k for faster retrieval)
                    retriever = agent_instance.vector_store.as_retriever(search_kwargs={"k": 3})
                    agent_instance.rag_chain = (
                        {"context": retriever | agent_instance._format_docs, "question": RunnablePassthrough()}
                        | agent_instance._create_rag_prompt()
                        | agent_instance.llm
                        | StrOutputParser()
                    )
                    print("RAG chain recreated successfully")
                except Exception as e:
                    print(f"Error loading vector store: {e}. Re-ingesting documents...")
                    agent_instance.ingest_documents()
            
            # Create the agent
            agent_instance.create_agent()
            print("Agent initialized successfully!")
        except Exception as e:
            import traceback
            print(f"Error initializing agent: {e}")
            print(traceback.format_exc())
            raise
    
    return agent_instance


def search_by_contract_id(contract_id: str):
    """Search for information by contract ID - direct search for faster results."""
    if not contract_id or not contract_id.strip():
        return "Please enter a contract ID (e.g., HD-7961)", ""
    
    try:
        # Ensure agent is initialized
        agent = initialize_agent()
        
        # Double-check that vector store is initialized
        if agent.vector_store is None:
            print("Vector store is None, attempting to initialize...")
            # Try to load or recreate
            if Path(vector_db_path).exists():
                from langchain_openai import OpenAIEmbeddings
                from langchain_community.vectorstores import Chroma
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    embeddings = OpenAIEmbeddings(api_key=api_key)
                    agent.vector_store = Chroma(
                        persist_directory=vector_db_path,
                        embedding_function=embeddings
                    )
                    print("Vector store loaded successfully")
            else:
                print("Vector store not found, ingesting documents...")
                agent.ingest_documents()
        
        # Use direct search method for faster, more reliable results
        contract_id_clean = contract_id.strip()
        result = agent.search_by_contract_id(contract_id_clean)
        
        # Format the result nicely
        if result and result.strip() and not result.startswith("Error"):
            # Check if it's the "doesn't exist" message
            if "doesn't exist" in result.lower():
                # Return the message as-is
                return result, result
            else:
                # Format actual results
                formatted_result = f"Contract ID: {contract_id_clean}\n{'='*50}\n\n{result}"
                return formatted_result, result  # Return both formatted and raw for expand
        else:
            # Return error message or default message
            if result and result.startswith("Error"):
                return result, result
            else:
                # No results found for this contract ID
                return f"Contract ID {contract_id_clean} doesn't exist.", f"Contract ID {contract_id_clean} doesn't exist."
            
    except Exception as e:
        import traceback
        error_msg = f"Error searching for contract ID: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  # Log to console for debugging
        error_response = f"Error searching for contract ID: {str(e)}. Please check the console for details."
        return error_response, error_response


def general_query(query: str, show_steps: bool = True):
    """Handle general queries using the agent with multi-step chaining support.
    
    Args:
        query: The user's query
        show_steps: Whether to show intermediate reasoning steps
    """
    if not query or not query.strip():
        return "Please enter a query or question."
    
    try:
        # Ensure agent is initialized
        agent = initialize_agent()
        
        # Run the query through the agent with multi-step reasoning
        # The agent will automatically chain multiple tool calls if needed
        result = agent.run(query.strip(), show_intermediate_steps=show_steps)
        
        return result
    except Exception as e:
        import traceback
        error_msg = f"Error processing query: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  # Log to console for debugging
        return f"Error processing query: {str(e)}. Please check the console for details."


def reload_documents():
    """Reload/reingest documents into the vector database."""
    try:
        import time
        import gc
        from pathlib import Path
        
        global agent_instance, vector_db_path
        
        # Step 1: Close and clear existing agent
        if agent_instance is not None:
            # Clear vector store reference
            if hasattr(agent_instance, 'vector_store'):
                agent_instance.vector_store = None
            if hasattr(agent_instance, 'rag_chain'):
                agent_instance.rag_chain = None
            if hasattr(agent_instance, 'agent_executor'):
                agent_instance.agent_executor = None
            
            # Reset agent
            agent_instance = None
            
            # Force garbage collection to release file handles
            gc.collect()
            time.sleep(1)
        
        # Step 2: Use a NEW path to avoid readonly database issues
        # This is the safest approach - create a fresh database instead of trying to delete the old one
        import time as time_module
        import uuid
        new_db_path = f"./agent_chroma_db_{int(time_module.time())}"
        
        print(f"Creating new database at: {new_db_path}")
        
        # Step 3: Create new agent instance with NEW path and ingest documents
        agent_instance = Agent(pdf_url=pdf_url, vector_db_path=new_db_path)
        agent_instance.ingest_documents()
        agent_instance.create_agent()
        
        # Step 4: Update the global path to point to the new database
        vector_db_path = new_db_path
        
        # Step 5: Try to clean up old database (non-blocking, won't fail if locked)
        try:
            old_path = Path("./agent_chroma_db")
            if old_path.exists() and str(old_path) != new_db_path:
                # Try to rename old database for later cleanup
                try:
                    import shutil
                    old_backup = f"./agent_chroma_db_old_{int(time_module.time())}"
                    old_path.rename(old_backup)
                    print(f"Moved old database to {old_backup} for cleanup")
                except Exception:
                    # If rename fails, it's okay - we're using new path anyway
                    print(f"Note: Old database at {old_path} may still exist but won't be used")
        except Exception as cleanup_error:
            # Don't fail if cleanup doesn't work
            print(f"Note: Could not cleanup old database: {cleanup_error}")
        
        return f"Documents reloaded successfully! New database created at: {new_db_path}"
    except Exception as e:
        import traceback
        error_msg = f"Error reloading documents: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error reloading documents: {str(e)}. Please check the console for details."


def reload_documents_with_new_path(new_path: str):
    """Helper function to reload with a new database path."""
    global agent_instance, vector_db_path
    
    try:
        # Update the global path
        old_path = vector_db_path
        vector_db_path = new_path
        
        # Reset agent
        agent_instance = None
        import gc
        gc.collect()
        import time
        time.sleep(1)
        
        # Create new agent with new path
        agent_instance = Agent(pdf_url=pdf_url, vector_db_path=vector_db_path)
        agent_instance.ingest_documents()
        agent_instance.create_agent()
        
        return f"Documents reloaded successfully! Using new database path: {new_path}. Note: The old database at {old_path} may need to be manually deleted later."
    except Exception as e:
        import traceback
        error_msg = f"Error reloading with new path: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error reloading documents: {str(e)}. Please check the console for details."


# Initialize agent on app startup
print("Pre-initializing agent on startup...")
try:
    initialize_agent()
    print("✅ Agent pre-initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-initialize agent: {e}")
    print("Agent will be initialized on first use.")

# Create Gradio interface
with gr.Blocks(title="Agent - Document Search", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🤖 Agent System - Document Search
    
    Search for insurance contract information by Contract ID or ask general questions.
    
    **Features:**
    - 🔍 Search by Contract ID (e.g., HD-7961)
    - 📄 Query insurance documents
    - 🌐 Internet Search Capabilities
    """)
    
    with gr.Tabs():
        with gr.TabItem("Search by Contract ID"):
            gr.Markdown("### Enter a Contract ID to search for information")
            with gr.Row():
                contract_id_input = gr.Textbox(
                    label="Contract ID",
                    placeholder="e.g., HD-7961",
                    value="HD-7961",
                    scale=4
                )
                contract_search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
            
            contract_output = gr.Textbox(
                label="Search Results",
                lines=20,
                interactive=True,
                show_copy_button=True
            )
            
            # Add action buttons for results
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Results", variant="secondary", size="sm")
                expand_btn = gr.Button("📄 View Full Details", variant="secondary", size="sm")
            
            # Store for expand functionality
            full_results = gr.State(value="")
            
            contract_search_btn.click(
                fn=search_by_contract_id,
                inputs=contract_id_input,
                outputs=[contract_output, full_results]
            )
            
            contract_id_input.submit(
                fn=search_by_contract_id,
                inputs=contract_id_input,
                outputs=[contract_output, full_results]
            )
            
            def clear_results():
                return "", ""
            
            def expand_results(contract_id, current_results, stored_results):
                if not contract_id or not contract_id.strip():
                    return current_results, stored_results
                
                # If we have stored results, use them; otherwise search again
                if stored_results and len(stored_results) > len(current_results):
                    return stored_results, stored_results
                else:
                    # Get full details
                    agent = initialize_agent()
                    result = agent.search_by_contract_id(contract_id.strip())
                    return result, result
            
            clear_btn.click(
                fn=clear_results,
                inputs=None,
                outputs=[contract_output, full_results]
            )
            
            expand_btn.click(
                fn=expand_results,
                inputs=[contract_id_input, contract_output, full_results],
                outputs=[contract_output, full_results]
            )
        
        with gr.TabItem("General Query"):
            gr.Markdown("""
            ### Ask any question about insurance documents or general queries
            
            **Multi-step Reasoning Enabled**: The agent can chain multiple searches and tool calls to gather comprehensive information.
            
            **Examples:**
            - "Search the internet for insurance certificate requirements and compare with our contract HD-7961"
            - "Find recent news about insurance trends and summarize how they might affect our coverage"
            - "Search for information about liability limits and check what our documents say about HD-7961"
            """)
            
            general_query_input = gr.Textbox(
                label="Your Query",
                placeholder="e.g., Search the internet for insurance certificate requirements and compare with contract HD-7961",
                lines=3
            )
            
            with gr.Row():
                show_steps_checkbox = gr.Checkbox(
                    label="Show Reasoning Steps",
                    value=True,
                    info="Display intermediate tool calls and reasoning steps"
                )
                general_query_btn = gr.Button("🔍 Submit Query", variant="primary", scale=2)
            
            general_output = gr.Textbox(
                label="Response",
                lines=25,
                interactive=True,
                show_copy_button=True
            )
            
            with gr.Row():
                clear_general_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
            
            def run_general_query(query, show_steps):
                return general_query(query, show_steps)
            
            general_query_btn.click(
                fn=run_general_query,
                inputs=[general_query_input, show_steps_checkbox],
                outputs=general_output
            )
            
            general_query_input.submit(
                fn=run_general_query,
                inputs=[general_query_input, show_steps_checkbox],
                outputs=general_output
            )
            
            clear_general_btn.click(
                fn=lambda: "",
                inputs=None,
                outputs=general_output
            )
        
        with gr.TabItem("Admin"):
            gr.Markdown("### Administrative Functions")
            gr.Markdown("Use this to reload documents if the source PDF has been updated.")
            reload_btn = gr.Button("Reload Documents", variant="secondary")
            reload_output = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False
            )
            
            reload_btn.click(
                fn=reload_documents,
                inputs=None,
                outputs=reload_output
            )
    
    gr.Markdown("""
    ---
    **Note:** The system uses RAG (Retrieval-Augmented Generation) to search insurance documents.
    """)


if __name__ == "__main__":
    # Initialize agent on startup
    print("Starting Agent Web Interface...")
    print("=" * 60)
    
    # Check for required environment variables
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please set it in your .env file or environment.")
        print("=" * 60)
    
    # Launch the app with public URL sharing
    print("\n" + "="*60)
    print("Launching web interface with public URL...")
    print("="*60)
    print("The public URL will be displayed below and opened in your browser.")
    print("="*60 + "\n")
    
    try:
        # Launch with public URL sharing
        result = app.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,
            share=True,  # Create public URL
            inbrowser=True,
            show_error=True,
            quiet=False,
            share_server_address=None,  # Let Gradio generate the public URL
            share_server_protocol="https"
        )
        
        # Print the public URL if available
        if hasattr(result, 'share_url') and result.share_url:
            print("\n" + "="*60)
            print(f"🌐 PUBLIC URL: {result.share_url}")
            print("="*60)
            print("Share this URL with others to give them access to your app!")
            print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error launching with sharing: {e}")
        print("Trying without public URL sharing...")
        # Fallback: local only
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )

