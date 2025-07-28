#!/usr/bin/env python3
"""
Standalone Keyword Clustering API Server
Production-ready Flask API with Gunicorn/Waitress support
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

# Auto-install critical packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

# Install Flask if not available
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Installing Flask and Flask-CORS...")
    install_package("flask")
    install_package("flask-cors")
    from flask import Flask, request, jsonify
    from flask_cors import CORS

# Install sentence-transformers if not available
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("sentence-transformers not available. Please install:")
    print("pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Global model variable
model = None

def load_model_once():
    """Load the model once on startup"""
    global model
    if SENTENCE_TRANSFORMERS_AVAILABLE and model is None:
        try:
            print("Loading sentence-transformers model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    return model

def get_median_title_cluster(cluster, corpus_keywords):
    """Get representative title for cluster"""
    title_lens = [len(corpus_keywords[i]) for i in cluster]
    median_idx = cluster[np.argsort(title_lens)[len(title_lens) // 2]]
    return corpus_keywords[median_idx]

def perform_clustering_api(keywords, threshold_val, min_community_size_val):
    """Core clustering function for API"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or model is None:
        return None
    
    # Clean keywords
    corpus_keywords = list(set([k.strip() for k in keywords if k.strip()]))
    
    if not corpus_keywords:
        return None
    
    try:
        # Encode keywords
        start_time = time.time()
        corpus_embeddings = model.encode(corpus_keywords, convert_to_tensor=True)
        encoding_time = time.time() - start_time
        
        # Cluster
        start_clustering = time.time()
        clusters = util.community_detection(
            corpus_embeddings, 
            min_community_size=min_community_size_val, 
            threshold=threshold_val
        )
        clustering_time = time.time() - start_clustering
        
        # Prepare results
        clusters_data = []
        clustered_keywords = set()
        
        for i, cluster in enumerate(clusters):
            cluster_name = get_median_title_cluster(cluster, corpus_keywords)
            cluster_keywords = [corpus_keywords[keyword_id] for keyword_id in cluster]
            
            clusters_data.append({
                'Cluster ID': i + 1,
                'Cluster Name': cluster_name,
                'Keywords': ', '.join(cluster_keywords),
                'Keyword Count': len(cluster_keywords),
                'Keyword List': cluster_keywords
            })
            clustered_keywords.update(cluster_keywords)
        
        # Handle unclustered
        unclustered_keywords = [k for k in corpus_keywords if k not in clustered_keywords]
        if unclustered_keywords:
            clusters_data.append({
                'Cluster ID': 0,
                'Cluster Name': 'Unclustered',
                'Keywords': ', '.join(unclustered_keywords),
                'Keyword Count': len(unclustered_keywords),
                'Keyword List': unclustered_keywords
            })
        
        return {
            'total_keywords': len(corpus_keywords),
            'total_clusters': len(clusters),
            'unclustered_count': len(unclustered_keywords),
            'processing_time': {
                'encoding_time': round(encoding_time, 2),
                'clustering_time': round(clustering_time, 2),
                'total_time': round(encoding_time + clustering_time, 2)
            },
            'parameters': {
                'threshold': threshold_val,
                'min_community_size': min_community_size_val
            },
            'clusters': clusters_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Clustering error: {e}")
        return None

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "keyword-clustering-api",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
    })

@app.route('/cluster', methods=['POST'])
def cluster_keywords():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'keywords' not in data:
            return jsonify({"error": "Missing required field: keywords"}), 400
        
        keywords = data['keywords']
        if not isinstance(keywords, list) or len(keywords) == 0:
            return jsonify({"error": "keywords must be a non-empty list"}), 400
        
        threshold = data.get('threshold', 0.75)
        min_community_size = data.get('min_community_size', 2)
        
        if not (0.0 <= threshold <= 1.0):
            return jsonify({"error": "threshold must be between 0.0 and 1.0"}), 400
        
        if not (1 <= min_community_size <= 20):
            return jsonify({"error": "min_community_size must be between 1 and 20"}), 400
        
        result = perform_clustering_api(keywords, threshold, min_community_size)
        
        if result is None:
            return jsonify({"error": "Clustering failed"}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def api_info():
    return jsonify({
        "service": "Keyword Clustering API",
        "version": "1.0.0",
        "status": "production-ready",
        "endpoints": {
            "POST /cluster": "Cluster keywords",
            "GET /health": "Health check",
            "GET /": "API info"
        }
    })

if __name__ == '__main__':
    # Load model on startup
    load_model_once()
    
    # Get port from environment or default
    port = int(os.environ.get('PORT', 5000))
    
    # Production deployment strategy
    production_server_started = False
    
    # Try Waitress first (most compatible)
    try:
        from waitress import serve
        print(f"🚀 Starting Waitress production server on port {port}")
        print(f"📊 Clustering endpoint: http://localhost:{port}/cluster")
        print(f"🔍 Health check: http://localhost:{port}/health")
        print("🔒 Production-ready WSGI server")
        
        serve(
            app, 
            host='0.0.0.0', 
            port=port, 
            threads=4,
            connection_limit=1000,
            cleanup_interval=30,
            channel_timeout=120,
            max_request_body_size=10485760  # 10MB
        )
        production_server_started = True
        
    except ImportError:
        print("⚠️ Waitress not available, trying Gunicorn...")
        
        # Try Gunicorn (Unix/Linux systems)
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                             if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            options = {
                'bind': f'0.0.0.0:{port}',
                'workers': 2,
                'worker_class': 'sync',
                'timeout': 300,
                'keepalive': 2,
                'max_requests': 1000,
                'max_requests_jitter': 100
            }
            
            print(f"🚀 Starting Gunicorn production server on port {port}")
            print(f"📊 Clustering endpoint: http://localhost:{port}/cluster")
            print(f"🔍 Health check: http://localhost:{port}/health")
            print("🔒 Production-ready WSGI server with workers")
            
            StandaloneApplication(app, options).run()
            production_server_started = True
            
        except ImportError:
            print("⚠️ Gunicorn not available, falling back to Flask dev server...")
    
    # Fallback to Flask development server
    if not production_server_started:
        print(f"⚠️ Starting Flask development server on port {port}")
        print(f"📊 Clustering endpoint: http://localhost:{port}/cluster")
        print(f"🔍 Health check: http://localhost:{port}/health")
        print("⚠️ WARNING: Development server - install waitress or gunicorn for production")
        print("   pip install waitress  # Recommended for all platforms")
        print("   pip install gunicorn  # Unix/Linux only")
        
        app.run(
            host='0.0.0.0', 
            port=port, 
            debug=False, 
            threaded=True,
            use_reloader=False
        )
