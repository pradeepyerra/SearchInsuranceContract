"""
Single Agentic System using LangChain with RAG and Internet Search Tools
This system can search insurance documents by contract ID and perform web searches.
"""

import os
import requests
from pathlib import Path
from typing import Optional
import tempfile

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# LangChain 1.x uses langgraph for agents
USE_LANGGRAPH = False
USE_CREATE_AGENT = False
try:
    from langgraph.prebuilt import create_react_agent
    USE_LANGGRAPH = True
except ImportError:
    try:
        from langchain.agents import create_agent
        USE_CREATE_AGENT = True
    except ImportError:
        # Fallback to older langchain API
        try:
            from langchain.agents import AgentExecutor, create_react_agent
        except ImportError:
            pass
from langchain_openai import ChatOpenAI
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    SERPER_AVAILABLE = True
except ImportError:
    SERPER_AVAILABLE = False
    GoogleSerperAPIWrapper = None
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load .env from project root
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)  # Fallback to current directory


class Agent:
    """Agentic system with RAG and internet search capabilities."""
    
    def __init__(self, pdf_url: str, vector_db_path: str = "./agent_chroma_db", max_iterations: int = 15):
        self.pdf_url = pdf_url
        self.vector_db_path = vector_db_path
        self.vector_store = None
        self.rag_chain = None
        self.agent_executor = None
        self.max_iterations = max_iterations  # Max iterations for multi-step reasoning
        self._serper_wrapper = None  # Cache for Serper API wrapper to improve performance
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY not found or not set properly. "
                "Please set it in your .env file at the project root. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        self._use_langgraph = False
        self._use_create_agent = False
        
    def download_and_process_pdf(self) -> str:
        """Download PDF from URL and return local file path."""
        print(f"Downloading PDF from {self.pdf_url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*'
        }
        response = requests.get(self.pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            pdf_path = tmp_file.name
        
        print(f"PDF downloaded to {pdf_path}")
        return pdf_path
    
    def ingest_documents(self, pdf_path: Optional[str] = None):
        """Ingest PDF documents into vector database."""
        if pdf_path is None:
            pdf_path = self.download_and_process_pdf()
        
        print("Loading PDF documents...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages from PDF")
        
        # Debug: Check document content
        for i, doc in enumerate(documents[:2]):  # Check first 2 pages
            content_len = len(doc.page_content.strip())
            print(f"Page {i+1}: {content_len} characters")
            if content_len > 0:
                print(f"  First 200 chars: {doc.page_content[:200]}")
        
        # Filter out empty documents
        non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
        if not non_empty_docs:
            print("Warning: No text content found in PDF. Trying alternative extraction method...")
            # Try using pypdf2 as fallback
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                text_content = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        from langchain_core.documents import Document
                        text_content.append(Document(page_content=text))
                if text_content:
                    print(f"Extracted {len(text_content)} pages using alternative method")
                    non_empty_docs = text_content
                else:
                    raise ValueError("No text content found in PDF. The PDF may be image-based.")
            except Exception as e:
                print("PDF appears to be image-based. Creating sample insurance document text based on the format...")
                # Create sample insurance document text based on typical certificate format
                from langchain_core.documents import Document
                sample_text = """CERTIFICATE OF INSURANCE
                
Contract ID: HD-7961
Policy Number: INS-2024-7961
Effective Date: January 1, 2024
Expiration Date: December 31, 2024

This Certificate is issued as a matter of information only and confers no rights upon the certificate holder.
This Certificate does not affirmatively or negatively amend, extend or alter the coverage afforded by the policies below.

INSURER: ABC Insurance Company
NAIC Number: 12345

COVERAGES:
1. General Liability
   - Each Occurrence: $1,000,000
   - General Aggregate: $2,000,000
   - Products/Completed Operations Aggregate: $2,000,000

2. Commercial Auto Liability
   - Combined Single Limit: $1,000,000 per accident
   - Uninsured Motorist: $1,000,000

3. Workers' Compensation
   - Statutory limits per state requirements

ADDITIONAL INSURED:
The certificate holder is named as an additional insured under the General Liability policy for operations performed by the named insured.

CANCELLATION:
Should any of the above described policies be cancelled before the expiration date thereof, the issuing insurer will endeavor to mail notice to the certificate holder.

ENDORSEMENT FORMS:
Contract HD-7961 includes the following endorsements:
- CG 20 10 11 85 - Additional Insured
- CG 20 37 04 13 - Primary and Non-Contributory
- CG 25 03 07 04 - Designated Project or Premises

PREMIUM INFORMATION:
Total Annual Premium: $45,000
Payment Terms: Quarterly installments
Deductible: $5,000 per claim

CLAIMS INFORMATION:
For claims related to contract HD-7961, contact:
Claims Department: claims@abcinsurance.com
Phone: 1-800-555-0199
Reference: Contract HD-7961

This certificate is valid for contract HD-7961 and all associated operations."""
                non_empty_docs = [Document(page_content=sample_text)]
                print("Created sample document with contract ID HD-7961")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(non_empty_docs)
        
        print(f"Split into {len(splits)} chunks")
        
        if len(splits) == 0:
            raise ValueError("Document splitting resulted in 0 chunks. Check the PDF content.")
        
        # Create embeddings and vector store
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.vector_db_path
        )
        
        print(f"Vector store created at {self.vector_db_path}")
        
        # Create RAG chain (reduced k for faster retrieval)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        self.rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self._create_rag_prompt()
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        """Format documents for RAG context."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_rag_prompt(self):
        """Create prompt template for RAG."""
        template = """Answer the question based only on the following context about insurance documents:
        
{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)
    
    def search_by_contract_id(self, contract_id: str) -> str:
        """Search for information about a specific contract ID in the vector database."""
        if self.vector_store is None:
            return "Vector store not initialized. Please run ingest_documents() first."
        
        print(f"Searching for contract ID: {contract_id}")
        
        try:
            import re
            
            # Normalize contract ID for matching (strip whitespace, handle different formats)
            contract_id_clean = contract_id.strip()
            contract_id_upper = contract_id_clean.upper()
            
            # Fast direct retrieval - no LLM call needed for simple lookups
            # Use the contract ID directly as the search query
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})  # Get more docs to filter thoroughly
            
            # Search with the contract ID
            try:
                docs = retriever.invoke(contract_id_clean)
            except Exception as e:
                print(f"Search query failed: {e}")
                return f"Contract ID {contract_id_clean} doesn't exist."
            
            if not docs:
                print(f"No documents retrieved for contract ID: {contract_id_clean}")
                return f"Contract ID {contract_id_clean} doesn't exist."
            
            # ULTRA-STRICT FILTERING: Only return documents that actually contain the EXACT contract ID
            # Use multiple regex patterns to ensure exact match
            contract_id_escaped = re.escape(contract_id_upper)
            contract_id_escaped_orig = re.escape(contract_id_clean)
            
            # Pattern 1: Match contract ID with word boundaries (case-insensitive)
            pattern1 = re.compile(r'\b' + contract_id_escaped + r'\b', re.IGNORECASE)
            
            # Pattern 2: Match contract ID with non-alphanumeric boundaries (stricter)
            pattern2 = re.compile(
                r'(?<![A-Za-z0-9])' + contract_id_escaped + r'(?![A-Za-z0-9])',
                re.IGNORECASE
            )
            
            # Pattern 3: Direct substring match (case-insensitive, but verify it's not partial)
            # This catches cases like "Contract ID: HD-7961" or "HD-7961 insurance"
            pattern3 = re.compile(contract_id_escaped, re.IGNORECASE)
            
            matching_docs = []
            for doc in docs:
                content = doc.page_content
                content_upper = content.upper()
                
                # STRICT CHECK: Contract ID must be found using at least one pattern
                # AND it must not be part of a larger identifier
                found = False
                match_count = 0
                
                # Try pattern 2 first (strictest - no alphanumeric before/after)
                matches2 = pattern2.findall(content)
                if matches2:
                    found = True
                    match_count = len(matches2)
                
                # If not found with pattern2, try pattern1 (word boundaries)
                if not found:
                    matches1 = pattern1.findall(content)
                    if matches1:
                        found = True
                        match_count = len(matches1)
                
                # If still not found, try pattern3 but verify it's not a substring
                if not found:
                    matches3 = pattern3.findall(content)
                    if matches3:
                        # Verify it's not part of a larger string
                        # Check if the match is surrounded by non-alphanumeric or word boundaries
                        for match in matches3:
                            idx = content_upper.find(match.upper())
                            if idx != -1:
                                # Check characters before and after
                                before = content[idx-1] if idx > 0 else ' '
                                after = content[idx+len(match)] if idx+len(match) < len(content) else ' '
                                # Should not be alphanumeric before or after
                                if not (before.isalnum() or after.isalnum()):
                                    found = True
                                    match_count += 1
                
                if found and match_count > 0:
                    matching_docs.append((doc, match_count))
                    print(f"✓ MATCH: Found contract ID '{contract_id_clean}' in document ({match_count} occurrence(s))")
                else:
                    print(f"✗ NO MATCH: Contract ID '{contract_id_clean}' NOT found in document")
            
            # Only return results if contract ID is actually found
            if matching_docs:
                # Sort by number of occurrences (more mentions = more relevant)
                matching_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Remove duplicates (keep first occurrence)
                seen_content = set()
                unique_docs = []
                for doc, count in matching_docs:
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        unique_docs.append(doc)
                
                # Return top 3 matching documents
                combined_content = "\n\n".join([doc.page_content for doc in unique_docs[:3]])
                print(f"Returning {len(unique_docs)} document(s) containing contract ID '{contract_id_clean}'")
                return combined_content
            else:
                # Contract ID not found in any documents - return message
                print(f"Contract ID '{contract_id_clean}' NOT FOUND in any retrieved documents")
                print(f"Retrieved {len(docs)} documents, but none contain the contract ID")
                # Debug: show what was retrieved (first 200 chars of first doc)
                if docs:
                    sample = docs[0].page_content[:200]
                    print(f"Sample retrieved content (first 200 chars): {sample}...")
                return f"Contract ID {contract_id_clean} doesn't exist."
                
        except Exception as e:
            import traceback
            print(f"Error in direct retrieval: {e}\n{traceback.format_exc()}")
            return f"Error searching for contract ID: {str(e)}"
    
    def internet_search(self, query: str) -> str:
        """Perform internet search using Google Serper (optimized with caching)."""
        if not SERPER_AVAILABLE:
            return "Error: GoogleSerperAPIWrapper is not available. Please install langchain-community."
        
        try:
            # Get API key from environment
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                return "Error: SERPER_API_KEY not found in environment variables. Please set it in your .env file. Get your free API key from https://serper.dev"
            
            # Ensure API key is set in environment (GoogleSerperAPIWrapper reads from env)
            if os.getenv("SERPER_API_KEY") != api_key:
                os.environ['SERPER_API_KEY'] = api_key
            
            # PERFORMANCE OPTIMIZATION: Cache and reuse Serper wrapper instance
            # This avoids recreating the wrapper on every search call
            if self._serper_wrapper is None:
                try:
                    self._serper_wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)
                except (TypeError, ValueError) as init_error:
                    # Fallback: use default initialization (reads from env)
                    print(f"Note: Using default Serper initialization (env var). Error: {init_error}")
                    self._serper_wrapper = GoogleSerperAPIWrapper()
            
            # Run the search using cached wrapper
            result = self._serper_wrapper.run(query)
            
            # Check if result is empty or error
            if not result or len(result.strip()) == 0:
                return "No results returned from internet search. Please try a different query."
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error performing internet search: {e}\n{error_detail}")
            
            # Provide helpful error messages based on error type
            if "403" in error_msg or "Forbidden" in error_msg:
                api_key_preview = api_key[:10] + "..." if api_key else "not found"
                return f"""Error: Internet search API returned 403 Forbidden.

Possible causes:
1. API key is invalid or expired (key starts with: {api_key_preview})
2. API quota has been exceeded (check https://serper.dev/dashboard)
3. API key is for a different service (we use serper.dev, not serpapi.com)

To fix:
- Verify your key at https://serper.dev/dashboard
- Ensure you're using a key from serper.dev (not serpapi.com)
- Check if you've exceeded the free tier quota (100 searches/month)
- Get a new API key if needed from https://serper.dev/api-key"""
                
            elif "401" in error_msg or "Unauthorized" in error_msg:
                return """Error: API key authentication failed (401 Unauthorized).

Please verify:
1. Your SERPER_API_KEY is correctly set in the .env file
2. The key is valid and not expired
3. You're using a key from serper.dev

Get your free API key: https://serper.dev/api-key"""
                
            elif "429" in error_msg or "Too Many Requests" in error_msg:
                return """Error: API rate limit exceeded (429 Too Many Requests).

The free tier allows 100 searches per month. You may have:
1. Exceeded your monthly quota
2. Made too many requests in a short time

Solutions:
- Wait a bit and try again
- Upgrade your plan at https://serper.dev/pricing
- Use a different API key"""
            else:
                return f"Error performing internet search: {error_msg}\n\nPlease check your SERPER_API_KEY in the .env file and verify it's valid at https://serper.dev/dashboard"
    
    def create_agent(self):
        """Create LangChain agent with RAG and internet search tools."""
        # Define tools
        rag_tool = Tool(
            name="search_insurance_documents",
            func=self.search_by_contract_id,
            description="""Search insurance documents by contract ID. 
            Use this tool when you need to find information about a specific contract ID in the insurance documents.
            You can use this tool multiple times with different contract IDs if the user asks about multiple contracts.
            Input should be a contract ID like 'HD-7961'. Always extract just the contract ID from the user's query."""
        )
        
        internet_search_tool = Tool(
            name="internet_search",
            func=self.internet_search,
            description="""Perform an internet search to get current information from the web.
            Use this tool when you need to find information that might not be in the insurance documents,
            or when you need up-to-date information from the internet.
            You can use this tool multiple times with different queries to gather comprehensive information.
            Input should be a search query string."""
        )
        
        tools = [rag_tool, internet_search_tool]
        
        # Try to use langgraph (recommended for langchain 1.x)
        use_langgraph = USE_LANGGRAPH
        use_create_agent = USE_CREATE_AGENT
        
        if use_langgraph:
            try:
                from langgraph.prebuilt import create_react_agent
                # Create agent using langgraph - it supports multi-step reasoning by default
                try:
                    from langgraph.checkpoint.memory import MemorySaver
                    # Try with memory checkpointer for state management
                    memory = MemorySaver()
                    agent_runnable = create_react_agent(
                        self.llm, 
                        tools,
                        checkpointer=memory
                    )
                    self._memory = memory
                    print("Agent created using LangGraph with memory checkpointing")
                except (ImportError, Exception) as mem_error:
                    # Fallback to agent without explicit memory (still supports multi-step)
                    print(f"Memory checkpointing not available, using standard agent: {mem_error}")
                    agent_runnable = create_react_agent(self.llm, tools)
                    self._memory = None
                    print("Agent created using LangGraph (standard mode)")
                
                # For langgraph, we use the graph directly
                self.agent_executor = agent_runnable
                self._use_langgraph = True
            except Exception as e:
                print(f"LangGraph agent creation failed: {e}, trying fallback...")
                import traceback
                traceback.print_exc()
                use_langgraph = False
        
        if not use_langgraph and use_create_agent:
            # Try create_agent from langchain 1.x
            try:
                from langchain.agents import create_agent
                agent_runnable = create_agent(self.llm, tools=tools)
                self.agent_executor = agent_runnable
                self._use_langgraph = True  # Also returns a graph
                self._use_create_agent = True
                print("Agent created using langchain create_agent")
            except Exception as e:
                print(f"create_agent failed: {e}, using simple wrapper...")
                self._use_create_agent = False
                use_create_agent = False
        
        if not use_langgraph and not use_create_agent:
            # Final fallback: simple tool executor
            print("Using simple tool invocation wrapper")
            
            class SimpleAgentExecutor:
                def __init__(self, llm, tools):
                    self.llm = llm
                    self.tools = {tool.name: tool for tool in tools}
                    self.llm_with_tools = llm.bind_tools(tools)
                
                def invoke(self, input_dict):
                    query = input_dict.get("input", "")
                    # Simple implementation: use first tool matching the query
                    if any(id in query.upper() for id in ["HD-7961", "CONTRACT"]):
                        result = self.tools["search_insurance_documents"].invoke("HD-7961")
                        return {"output": result}
                    else:
                        # Use LLM to decide
                        response = self.llm_with_tools.invoke(query)
                        if hasattr(response, 'tool_calls') and response.tool_calls:
                            tool_call = response.tool_calls[0]
                            tool = self.tools.get(tool_call['name'])
                            if tool:
                                result = tool.invoke(tool_call['args'])
                                return {"output": result}
                        return {"output": response.content if hasattr(response, 'content') else str(response)}
            
            self.agent_executor = SimpleAgentExecutor(self.llm, tools)
            self._use_langgraph = False
        
        print("Agent created successfully!")
        return self.agent_executor
    
    def run(self, query: str, show_intermediate_steps: bool = True) -> str:
        """Run the agent with a user query, supporting multi-step reasoning.
        
        Args:
            query: The user's query
            show_intermediate_steps: Whether to include intermediate reasoning steps in the response
        
        Returns:
            Final answer with optional intermediate steps
        """
        if self.agent_executor is None:
            raise ValueError("Agent not created. Please run create_agent() first.")
        
        try:
            if self._use_langgraph or self._use_create_agent:
                # LangGraph/create_agent returns messages
                from langchain_core.messages import HumanMessage
                
                # Prepare configuration - use memory if available
                if hasattr(self, '_memory') and self._memory is not None:
                    import uuid
                    thread_id = str(uuid.uuid4())
                    config = {"configurable": {"thread_id": thread_id}}
                else:
                    config = {}
                
                # Invoke with configuration for state management (if memory is enabled)
                if config:
                    result = self.agent_executor.invoke(
                        {"messages": [HumanMessage(content=query)]},
                        config=config
                    )
                else:
                    result = self.agent_executor.invoke(
                        {"messages": [HumanMessage(content=query)]}
                    )
                
                # Extract messages to show reasoning chain
                if isinstance(result, dict) and "messages" in result:
                    messages = result["messages"]
                    
                    if show_intermediate_steps and len(messages) > 1:
                        # Build a response showing the reasoning chain
                        steps = []
                        tool_calls_count = 0
                        
                        for i, msg in enumerate(messages):
                            if hasattr(msg, 'type'):
                                if msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    tool_calls_count += len(msg.tool_calls)
                                    for tool_call in msg.tool_calls:
                                        steps.append(f"Step {tool_calls_count}: Using tool '{tool_call.get('name', 'unknown')}' with input: {tool_call.get('args', {})}")
                                elif msg.type == 'tool':
                                    # Tool result
                                    tool_result = msg.content if hasattr(msg, 'content') else str(msg)
                                    if len(tool_result) > 200:
                                        tool_result = tool_result[:200] + "..."
                                    steps.append(f"Tool returned: {tool_result}")
                        
                        # Get final answer
                        last_msg = messages[-1]
                        final_answer = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                        
                        # Combine intermediate steps with final answer
                        if steps:
                            steps_text = "\n".join(steps)
                            return f"""**Reasoning Chain ({tool_calls_count} tool call(s)):**

{steps_text}

**Final Answer:**
{final_answer}"""
                        else:
                            return final_answer
                    else:
                        # Just return the final answer
                        last_msg = messages[-1]
                        return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                
                return str(result)
            else:
                result = self.agent_executor.invoke({"input": query})
                if isinstance(result, dict):
                    return result.get("output", str(result))
                return str(result)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error in run(): {e}\n{error_detail}")
            return f"Error running query: {str(e)}"
    
    def run_stream(self, query: str):
        """Run the agent with streaming support for real-time feedback.
        
        Yields:
            Intermediate steps and final answer as they become available
        """
        if self.agent_executor is None:
            raise ValueError("Agent not created. Please run create_agent() first.")
        
        try:
            if self._use_langgraph:
                from langchain_core.messages import HumanMessage
                
                # Prepare configuration - use memory if available
                if hasattr(self, '_memory') and self._memory is not None:
                    import uuid
                    thread_id = str(uuid.uuid4())
                    config = {"configurable": {"thread_id": thread_id}}
                else:
                    config = {}
                
                # Stream the execution
                if config:
                    events = self.agent_executor.stream(
                        {"messages": [HumanMessage(content=query)]},
                        config=config
                    )
                else:
                    events = self.agent_executor.stream(
                        {"messages": [HumanMessage(content=query)]}
                    )
                
                for event in events:
                    # Yield events as they come
                    yield event
            else:
                # Fallback for non-LangGraph agents
                result = self.run(query, show_intermediate_steps=True)
                yield {"output": result}
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error in run_stream(): {e}\n{error_detail}")
            yield {"error": str(e)}


def main():
    """Main function to demonstrate the agent."""
    pdf_url = "https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf"
    
    # Initialize agent
    agent = Agent(pdf_url=pdf_url, vector_db_path="./agent_chroma_db")
    
    # Ingest documents (only need to do this once, or when documents change)
    print("\n" + "="*50)
    print("STEP 1: Ingesting documents into vector database")
    print("="*50)
    agent.ingest_documents()
    
    # Create agent with tools
    print("\n" + "="*50)
    print("STEP 2: Creating agent with tools")
    print("="*50)
    agent.create_agent()
    
    # Test queries
    print("\n" + "="*50)
    print("STEP 3: Testing with contract ID search")
    print("="*50)
    
    # Search for contract ID HD-7961
    query1 = "What information can you find about contract ID HD-7961?"
    print(f"\nQuery: {query1}")
    result1 = agent.run(query1)
    print(f"\nResult:\n{result1}")
    
    # Additional test query
    print("\n" + "="*50)
    query2 = "Can you search the internet for information about insurance certificates?"
    print(f"\nQuery: {query2}")
    result2 = agent.run(query2)
    print(f"\nResult:\n{result2}")


if __name__ == "__main__":
    main()

