"""
Flask routes module.
Defines all REST API endpoints for the Knowledge Store application.
"""

import os
import json
from flask import Blueprint, request, jsonify, stream_with_context, Response
from werkzeug.utils import secure_filename
from langchain_core.messages import HumanMessage, AIMessage

from .config import config
from .rag_engine import RAGEngine
from .llm import LLMManager
from .query_classifier import QueryClassifier
from .answer_extractor import AnswerExtractor
from .graph_rag import get_graph_rag


# Create blueprint
api = Blueprint('api', __name__)

# Initialize components
rag = RAGEngine()
llm = LLMManager()


@api.route("/status", methods=["GET"])
def status():
    """Get system status."""
    return jsonify(rag.get_status())


@api.route("/ingest/folder", methods=["POST"])
def ingest_folder():
    """Ingest all .txt files from the default FAQ folder."""
    result = rag.ingest_folder(str(config.UPLOAD_FOLDER))
    return jsonify(result)


@api.route("/upload", methods=["POST"])
def upload_file():
    """Upload one or more .txt files and ingest them immediately."""
    if "files" not in request.files:
        return jsonify({"status": "error", "message": "No files part in request."}), 400
    
    files = request.files.getlist("files")
    results = []
    
    for f in files:
        if f.filename == "":
            continue
        if not f.filename.endswith(".txt"):
            results.append({"file": f.filename, "status": "skipped", "reason": "not a .txt file"})
            continue
        
        filename = secure_filename(f.filename)
        save_path = config.UPLOAD_FOLDER / filename
        config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        f.save(str(save_path))
        
        result = rag.ingest_file(str(save_path))
        result["file"] = filename
        results.append(result)
    
    return jsonify({"uploads": results})


@api.route("/clear", methods=["POST"])
def clear_index():
    """Clear all indexed data."""
    result = rag.clear_index()
    return jsonify(result)


@api.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint - uses RAG + LLM for intelligent responses.
    Body: { "message": "...", "history": [ {role, content}, ... ] }
    Returns a streaming plain-text response.
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Convert history from dicts to BaseMessage objects if provided
    history_data = data.get("history", [])
    history = []
    for msg in history_data:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    
    # Classify query to determine processing strategy
    is_simple, simple_response = QueryClassifier.is_simple_query(user_message)
    
    if is_simple:
        sources = []
    else:
        # Use the self-correcting graph for complex queries (with history)
        graph_rag = get_graph_rag()
        
        # Use threading-based timeout for cross-platform compatibility
        import threading
        
        result_container = [None]
        error_container = [None]
        
        def run_query():
            try:
                result_container[0] = graph_rag.query(user_message, history=history if history else None)
            except Exception as e:
                error_container[0] = str(e)
        
        thread = threading.Thread(target=run_query)
        thread.daemon = True
        thread.start()
        thread.join(timeout=15)  # 15 second timeout to prevent hanging
        
        if thread.is_alive():
            # Thread is still running - query is taking too long
            # Fallback to direct RAG retrieval without LLM
            chunks = rag.retrieve(user_message, top_k=3)
            extracted = AnswerExtractor.extract(user_message, chunks) if chunks else ""
            if extracted:
                result = {"answer": extracted, "sources": list({c["source"] for c in chunks if chunks})}
            else:
                result = {"answer": "I apologize, but the query took too long to process. Please try again.", "sources": [], "quality_score": 0}
        elif error_container[0]:
            print(f"[Routes] Graph query error: {error_container[0]}")
            result = {"answer": "An error occurred while processing your query. Please try again.", "sources": [], "quality_score": 0}
        else:
            result = result_container[0]
        
        sources = result.get("sources", [])
        
        def generate():
            # Send sources metadata first
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            response_text = result.get("answer", "")
            
            if not response_text:
                response_text = "I don't have any relevant information in the knowledge base. Please upload FAQ documents using the sidebar, then try again."
            
            # Stream response
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    
    # For simple queries, use the original logic
    chunks = []
    
    def generate():
        # Send sources metadata first
        yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
        
        if simple_response:
            response_text = simple_response
        else:
            response_text = "I don't have any relevant information in the knowledge base. Please upload FAQ documents using the sidebar, then try again."
        
        # Stream response
        chunk_size = 50
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i+chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _generate_with_llm(user_message: str, chunks: list) -> str:
    """Generate response using LLM with retrieved context."""
    context_text = "\n\n".join(c["text"][:1000] for c in chunks[:3] if c["text"])
    prompt = llm.build_rag_prompt(context_text, user_message)

    response_text = llm.generate(prompt)

    if response_text:
        return response_text

    # Fallback if LLM generation fails
    answer = AnswerExtractor.extract(user_message, chunks)
    if answer:
        return answer

    return AnswerExtractor.format_chunks_response(chunks)


def _generate_without_llm(user_message: str, chunks: list) -> str:
    """Generate response using extraction only (no LLM)."""
    answer = AnswerExtractor.extract(user_message, chunks)

    if answer:
        return answer

    return AnswerExtractor.format_chunks_response(chunks)
