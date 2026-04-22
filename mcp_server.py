"""
MCP Server for knowledgeStore RAG System.

This server exposes knowledgeStore functionality as MCP tools,
allowing MCP-compatible clients to query the RAG system.

Usage:
    python mcp_server.py

Requirements:
    pip install mcp

Configuration for Claude Desktop:
    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "knowledgeStore": {
          "command": "python",
          "args": ["path/to/mcp_server.py"]
        }
      }
    }
"""

import json
import sys
import os
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.rag_engine import RAGEngine
from src.llm import LocalLLM
from src.graph_rag import GraphRAG

# MCP SDK import
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Define placeholder types for type checking when MCP is not installed
    class Server:
        def __init__(self, name: str): pass
        def list_tools(self): pass
        def call_tool(self): pass
    class Tool:
        def __init__(self, name: str = "", description: str = "", inputSchema: dict = None): pass
    class TextContent:
        def __init__(self, type: str = "text", text: str = ""): pass
    print("Warning: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)


class KnowledgeStoreMCPServer:
    """MCP Server that exposes knowledgeStore RAG tools."""
    
    def __init__(self):
        """Initialize the RAG components."""
        self.rag_engine = None
        self.llm = None
        self.graph_rag = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of RAG components."""
        if not self._initialized:
            # Initialize RAG engine
            self.rag_engine = RAGEngine()
            
            # Initialize LLM
            self.llm = LocalLLM()
            
            # Initialize GraphRAG
            self.graph_rag = GraphRAG(self.rag_engine, self.llm)
            
            self._initialized = True
    
    def query_knowledge_base(self, question: str, history: Optional[list] = None) -> dict:
        """
        Query the knowledge base with a question.
        
        Args:
            question: The question to ask
            history: Optional conversation history
            
        Returns:
            dict with 'answer' and 'sources' keys
        """
        self._ensure_initialized()
        
        # Check if index exists
        if not self.rag_engine.index_exists():
            return {
                "answer": "The knowledge base is empty. Please ingest documents first.",
                "sources": [],
                "error": None
            }
        
        try:
            result = self.graph_rag.query(question, history=history or [])
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "error": None
            }
        except Exception as e:
            return {
                "answer": "",
                "sources": [],
                "error": str(e)
            }
    
    def get_status(self) -> dict:
        """
        Get knowledge base status.
        
        Returns:
            dict with document count, chunk count, etc.
        """
        self._ensure_initialized()
        
        status = {
            "index_exists": self.rag_engine.index_exists(),
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_results": config.TOP_K_RESULTS
        }
        
        if status["index_exists"]:
            metadata = self.rag_engine.get_metadata()
            status["document_count"] = metadata.get("document_count", 0)
            status["chunk_count"] = metadata.get("chunk_count", 0)
            status["sources"] = metadata.get("sources", [])
        else:
            status["document_count"] = 0
            status["chunk_count"] = 0
            status["sources"] = []
        
        return status
    
    def search_documents(self, query: str, top_k: int = 5) -> list:
        """
        Search documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        self._ensure_initialized()
        
        if not self.rag_engine.index_exists():
            return []
        
        try:
            results = self.rag_engine.retrieve(query, top_k=top_k)
            return [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            return []


def create_mcp_server() -> Server:
    """Create and configure the MCP server."""
    
    server = Server("knowledgeStore")
    knowledge_store = KnowledgeStoreMCPServer()
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available MCP tools."""
        return [
            Tool(
                name="query_knowledge_base",
                description="Query the knowledge base with a question. Returns an answer generated from relevant documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the knowledge base"
                        },
                        "history": {
                            "type": "array",
                            "description": "Optional conversation history for multi-turn dialogue",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["question"]
                }
            ),
            Tool(
                name="get_knowledge_base_status",
                description="Get the current status of the knowledge base including document count and sources.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="search_documents",
                description="Search for relevant documents without generating an answer. Returns the most relevant chunks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="ingest_documents",
                description="Ingest all .txt files from the data/faq_docs folder into the knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls from MCP clients."""
        
        if name == "query_knowledge_base":
            question = arguments.get("question", "")
            history = arguments.get("history")
            
            if not question:
                return [TextContent(type="text", text="Error: question is required")]
            
            result = knowledge_store.query_knowledge_base(question, history)
            
            if result.get("error"):
                return [TextContent(type="text", text=f"Error: {result['error']}")]
            
            response = f"Answer: {result['answer']}\n\nSources: {', '.join(result['sources'])}"
            return [TextContent(type="text", text=response)]
        
        elif name == "get_knowledge_base_status":
            status = knowledge_store.get_status()
            return [TextContent(type="text", text=json.dumps(status, indent=2))]
        
        elif name == "search_documents":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            
            if not query:
                return [TextContent(type="text", text="Error: query is required")]
            
            results = knowledge_store.search_documents(query, top_k)
            
            if not results:
                return [TextContent(type="text", text="No results found")]
            
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(f"[{i}] {r['text']}\n   Source: {r['source']} (score: {r['score']:.4f})")
            
            return [TextContent(type="text", text="\n\n".join(formatted))]
        
        elif name == "ingest_documents":
            # Import ingest function from src.routes
            try:
                from src.routes import rag
                result = rag.ingest_folder(str(config.UPLOAD_FOLDER))
                count = result.get("documents_ingested", 0)
                return [TextContent(type="text", text=f"Successfully ingested {count} documents")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error ingesting documents: {str(e)}")]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    return server


async def main():
    """Main entry point for the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed", file=sys.stderr)
        print("Run: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
