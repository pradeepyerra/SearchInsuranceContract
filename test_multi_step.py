#!/usr/bin/env python3
"""
Test script for multi-step reasoning functionality
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from agent import Agent

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

def test_multi_step_reasoning():
    """Test multi-step reasoning with chained tool calls."""
    
    print("="*70)
    print("Testing Multi-Step Reasoning and Chaining")
    print("="*70)
    
    # Initialize agent
    pdf_url = "https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf"
    agent = Agent(pdf_url=pdf_url, vector_db_path="./agent_chroma_db")
    
    # Load or create vector store
    if os.path.exists("./agent_chroma_db"):
        print("\n✓ Loading existing vector store...")
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        agent.vector_store = Chroma(
            persist_directory="./agent_chroma_db",
            embedding_function=embeddings
        )
    else:
        print("\n✓ Ingesting documents...")
        agent.ingest_documents()
    
    # Create agent with tools
    print("\n✓ Creating agent with multi-step reasoning support...")
    agent.create_agent()
    
    # Test queries that require multi-step reasoning
    test_queries = [
        {
            "query": "Search the internet for information about insurance certificates and tell me what you find",
            "description": "Single internet search"
        },
        {
            "query": "Search the internet for insurance certificate requirements and compare with information about contract HD-7961",
            "description": "Multi-step: Internet search + RAG search + comparison"
        },
        {
            "query": "Find information about contract HD-7961 and then search online for similar insurance coverage examples",
            "description": "Multi-step: RAG search + Internet search"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {test['description']}")
        print(f"{'='*70}")
        print(f"\nQuery: {test['query']}")
        print("\n" + "-"*70)
        print("RESPONSE (with intermediate steps):")
        print("-"*70)
        
        try:
            result = agent.run(test['query'], show_intermediate_steps=True)
            print(result)
            print("\n✓ Test completed successfully")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n")

if __name__ == "__main__":
    test_multi_step_reasoning()

