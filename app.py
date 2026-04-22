"""
Knowledge Store - Enterprise RAG FAQ System
Flask application entry point.
"""

import os
from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from src.config import config
from src.routes import api

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(api, url_prefix='/api')

# Ensure directories exist
config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


def auto_ingest():
    """Auto-ingest FAQ files on startup if available."""
    faq_files = [f for f in os.listdir(config.UPLOAD_FOLDER) if f.endswith((".txt", ".pdf", ".doc", ".docx"))]
    if faq_files:
        print(f"[App] Found {len(faq_files)} FAQ file(s) in {config.UPLOAD_FOLDER}. Auto-ingesting...")
        from src.routes import rag
        rag.ingest_folder(str(config.UPLOAD_FOLDER))


if __name__ == "__main__":
    print(f"[App] Starting Knowledge Store on http://localhost:{config.FLASK_PORT}")
    auto_ingest()
    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
        use_reloader=False
    )
