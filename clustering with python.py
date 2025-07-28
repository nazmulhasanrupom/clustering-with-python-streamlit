import subprocess
import sys
import importlib
import os

def install_and_import(package_name, import_name=None, fallback_packages=None):
    """Install and import a package if it's not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        # Check if we're on Streamlit Cloud - if so, skip installation attempts
        if (os.environ.get('STREAMLIT_CLOUD', False) or 
            '/home/adminuser' in os.getcwd() or
            '/mount/src' in os.getcwd() or
            os.environ.get('STREAMLIT_SHARING', False)):
            print(f"Streamlit Cloud detected - skipping installation of {package_name}")
            return False
        
        print(f"Installing {package_name}...")
        
        # Determine pip install command (avoid --user in virtual environments)
        pip_cmd = [sys.executable, "-m", "pip", "install"]
        
        # Check if we can use --user (not in virtual environment)
        try:
            import site
            if hasattr(site, 'USER_SITE') and site.USER_SITE:
                pip_cmd.append("--user")
        except:
            pass  # Skip --user if detection fails
        
        # Try main package first
        try:
            subprocess.check_call(pip_cmd + [package_name])
            importlib.import_module(import_name)
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            
            # Try fallback packages
            if fallback_packages:
                for fallback_cmd in fallback_packages:
                    try:
                        print(f"Trying fallback: {fallback_cmd}")
                        # Split the command properly
                        cmd_parts = pip_cmd + fallback_cmd.split()
                        subprocess.check_call(cmd_parts)
                        importlib.import_module(import_name)
                        return True
                    except Exception as fallback_error:
                        print(f"Fallback {fallback_cmd} also failed: {fallback_error}")
            
            return False

# Auto-install required packages with fallbacks
required_packages = [
    ("streamlit", "streamlit", None),
    ("pandas", "pandas", None),
    ("numpy", "numpy", None),
    ("requests", "requests", None),
    ("openpyxl", "openpyxl", None),
    ("sentence-transformers", "sentence_transformers", [
        "--no-deps sentence-transformers",
        "transformers torch numpy scikit-learn scipy tqdm"
    ]),
    ("flask", "flask", None),
    ("flask-cors", "flask_cors", None),
    ("gunicorn", "gunicorn", None),
    ("waitress", "waitress", None)
]

# Skip installation on Streamlit Cloud
STREAMLIT_CLOUD_DETECTED = (
    os.environ.get('STREAMLIT_CLOUD', False) or 
    '/home/adminuser' in os.getcwd() or
    '/mount/src' in os.getcwd() or
    os.environ.get('STREAMLIT_SHARING', False)
)

if not STREAMLIT_CLOUD_DETECTED:
    print("Checking and installing required packages...")
    failed_packages = []
    for package, import_name, fallbacks in required_packages:
        if not install_and_import(package, import_name, fallbacks):
            print(f"ERROR: Could not install {package}.")
            failed_packages.append(package)
            if package == "sentence-transformers":
                print("For sentence-transformers issues, try manual installation:")
                print("pip install --no-deps sentence-transformers")
                print("pip install transformers torch numpy scikit-learn scipy tqdm")

    # Continue even if some packages failed
    if failed_packages:
        print(f"Failed to install: {', '.join(failed_packages)}")
        print("Continuing with available packages...")
else:
    print("Streamlit Cloud detected - skipping automatic installation")
    print("Please ensure requirements.txt contains all necessary packages")

# Now import everything with fallback handling
import streamlit as st
import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import threading
import signal
import atexit

# Try to import Flask and related modules
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Try to import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check if we're running on Streamlit Cloud (limited installation permissions)
STREAMLIT_CLOUD = STREAMLIT_CLOUD_DETECTED

# ==================================================
# CORE CLUSTERING FUNCTIONS
# ==================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the sentence transformer model (cached for Streamlit)"""
    global SENTENCE_TRANSFORMERS_AVAILABLE
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        if STREAMLIT_CLOUD:
            st.error("❌ sentence-transformers not available on Streamlit Cloud without requirements.txt")
            st.markdown("""
            ### To fix this issue:
            
            1. Create a `requirements.txt` file in your repository:
            ```
            streamlit
            sentence-transformers
            pandas
            numpy
            requests
            openpyxl
            torch>=1.9.0
            transformers>=4.21.0
            flask
            flask-cors
            ```
            
            2. Redeploy your app with both files
            """)
            return None
        
        st.error("❌ sentence-transformers not available.")
        st.markdown("""
        ### Installation Required:
        
        This app requires sentence-transformers for AI clustering. Please install it manually:
        
        ```bash
        pip install sentence-transformers
        ```
        """)
        return None
    
    try:
        with st.spinner("🤖 Loading AI model... This may take up to 2 minutes on first run."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

def get_median_title_cluster(cluster, corpus_keywords):
    """Get the median length keyword from cluster as representative title"""
    title_lens = [len(corpus_keywords[i]) for i in cluster]
    median_idx = cluster[np.argsort(title_lens)[len(title_lens) // 2]]
    return corpus_keywords[median_idx]

def perform_clustering_core(keywords, threshold_val, min_community_size_val, use_streamlit=True):
    """Core clustering function that can work with or without Streamlit UI"""
    
    # Try to import sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer, util
        model_available = True
    except ImportError:
        model_available = False
    
    if not model_available:
        error_msg = "sentence-transformers not available. Please install it first."
        if use_streamlit:
            st.error(f"❌ {error_msg}")
        return None
    
    # Load model (without Streamlit caching for API)
    try:
        if use_streamlit:
            model = load_model()  # Use cached version
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        if use_streamlit:
            st.error(f"❌ {error_msg}")
        return None
    
    if model is None:
        return None
    
    # Clean and deduplicate keywords
    corpus_keywords = list(set([k.strip() for k in keywords if k.strip()]))
    
    if not corpus_keywords:
        error_msg = "No valid keywords found in the data."
        if use_streamlit:
            st.error(f"❌ {error_msg}")
        return None
    
    if len(corpus_keywords) < min_community_size_val:
        warning_msg = f"Only {len(corpus_keywords)} unique keywords found. Consider reducing minimum cluster size."
        if use_streamlit:
            st.warning(f"⚠️ {warning_msg}")
    
    # Show progress only for Streamlit
    if use_streamlit:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"🔤 Encoding {len(corpus_keywords)} unique keywords...")
        progress_bar.progress(25)
    
    try:
        # Encoding phase
        start_encoding = time.time()
        corpus_embeddings = model.encode(
            corpus_keywords, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_tensor=True
        )
        encoding_time = time.time() - start_encoding
        
        if use_streamlit:
            progress_bar.progress(50)
            status_text.text("🔍 Performing clustering analysis...")
        
        # Clustering phase
        start_clustering = time.time()
        clusters = util.community_detection(
            corpus_embeddings, 
            min_community_size=min_community_size_val, 
            threshold=threshold_val
        )
        clustering_time = time.time() - start_clustering
        
        if use_streamlit:
            progress_bar.progress(75)
            status_text.text("📊 Organizing results...")
        
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
        
        # Handle unclustered keywords
        unclustered_keywords = [k for k in corpus_keywords if k not in clustered_keywords]
        if unclustered_keywords:
            clusters_data.append({
                'Cluster ID': 0,
                'Cluster Name': 'Unclustered',
                'Keywords': ', '.join(unclustered_keywords),
                'Keyword Count': len(unclustered_keywords),
                'Keyword List': unclustered_keywords
            })
        
        if use_streamlit:
            progress_bar.progress(100)
            status_text.text("✅ Clustering completed!")
        
        # Create results summary
        result = {
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
        
        # Clear progress indicators for Streamlit
        if use_streamlit:
            progress_bar.empty()
            status_text.empty()
        
        return result
        
    except Exception as e:
        if use_streamlit:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during clustering: {str(e)}")
        return None

def perform_clustering(keywords, threshold_val, min_community_size_val):
    """Streamlit wrapper for the core clustering function"""
    return perform_clustering_core(keywords, threshold_val, min_community_size_val, use_streamlit=True)

def send_webhook_result(webhook_url, result_data):
    """Send the clustering result to the specified webhook URL"""
    try:
        with st.spinner(f"📤 Sending results to webhook..."):
            response = requests.post(
                webhook_url, 
                json=result_data, 
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            st.success(f"✅ Webhook sent successfully! Status: {response.status_code}")
            return True
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to send webhook: {str(e)}")
        return False

# ==================================================
# FLASK API FUNCTIONALITY
# ==================================================

# Global Flask app variable
flask_app = None
api_server_thread = None

def create_flask_app():
    """Create and configure the Flask API server"""
    if not FLASK_AVAILABLE:
        return None
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "service": "keyword-clustering-api",
            "timestamp": datetime.now().isoformat(),
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        })
    
    @app.route('/cluster', methods=['POST'])
    def cluster_keywords():
        """Main clustering endpoint for API requests"""
        try:
            # Validate content type
            if not request.is_json:
                return jsonify({
                    "error": "Content-Type must be application/json",
                    "received": request.content_type
                }), 400
            
            data = request.get_json()
            
            # Validate required fields
            if 'keywords' not in data:
                return jsonify({
                    "error": "Missing required field: keywords",
                    "required": ["keywords"],
                    "optional": ["threshold", "min_community_size", "webhook_url"]
                }), 400
            
            keywords = data['keywords']
            if not isinstance(keywords, list) or len(keywords) == 0:
                return jsonify({
                    "error": "keywords must be a non-empty list of strings"
                }), 400
            
            # Extract parameters with defaults
            threshold = data.get('threshold', 0.75)
            min_community_size = data.get('min_community_size', 2)
            webhook_url = data.get('webhook_url')
            
            # Validate parameters
            if not (0.0 <= threshold <= 1.0):
                return jsonify({
                    "error": "threshold must be between 0.0 and 1.0"
                }), 400
            
            if not (1 <= min_community_size <= 20):
                return jsonify({
                    "error": "min_community_size must be between 1 and 20"
                }), 400
            
            # Perform clustering (without Streamlit UI)
            result = perform_clustering_core(
                keywords, 
                threshold, 
                min_community_size, 
                use_streamlit=False
            )
            
            if result is None:
                return jsonify({
                    "error": "Clustering failed",
                    "details": "Model not available or clustering process failed"
                }), 500
            
            # Send webhook if provided
            webhook_sent = False
            webhook_error = None
            if webhook_url:
                try:
                    response = requests.post(
                        webhook_url, 
                        json=result, 
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    response.raise_for_status()
                    webhook_sent = True
                except Exception as e:
                    webhook_error = str(e)
            
            # Add webhook status to response
            result['webhook_status'] = {
                'sent': webhook_sent,
                'error': webhook_error,
                'url': webhook_url if webhook_url else None
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                "error": "Internal server error",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/', methods=['GET'])
    def api_info():
        """API information endpoint"""
        return jsonify({
            "service": "Keyword Clustering API",
            "version": "1.0.0",
            "endpoints": {
                "POST /cluster": "Cluster keywords using AI",
                "GET /health": "Health check",
                "GET /": "API information"
            },
            "example_request": {
                "url": "/cluster",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "keywords": ["machine learning", "AI", "data science"],
                    "threshold": 0.75,
                    "min_community_size": 2,
                    "webhook_url": "https://optional-webhook.com/endpoint"
                }
            },
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        })
    
    return app

def start_api_server(port=5000, production=True):
    """Start the Flask API server with production or development mode"""
    global flask_app, api_server_thread
    
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. API server cannot start.")
        return False
    
    try:
        flask_app = create_flask_app()
        if flask_app is None:
            return False
        
        # Check if we're on Streamlit Cloud or in a restricted environment
        is_streamlit_cloud = (
            os.environ.get('STREAMLIT_CLOUD', False) or 
            '/mount/src' in os.getcwd() or
            '/home/adminuser' in os.getcwd()
        )
        
        if production and not is_streamlit_cloud:
            # Try to use production servers (only for local environments)
            print("🚀 Starting production API server...")
            
            # Try Waitress first (cross-platform, thread-safe)
            try:
                from waitress import serve
                print("✅ Using Waitress production server")
                
                def run_waitress():
                    serve(
                        flask_app,
                        host='0.0.0.0',
                        port=port,
                        threads=4,
                        connection_limit=1000,
                        cleanup_interval=30,
                        channel_timeout=120
                    )
                
                api_server_thread = threading.Thread(target=run_waitress, daemon=True)
                api_server_thread.start()
                
                print(f"✅ Waitress API server started on http://0.0.0.0:{port}")
                print(f"📊 Clustering endpoint: http://localhost:{port}/cluster")
                print(f"🔍 Health check: http://localhost:{port}/health")
                print("🔒 Production-ready server with threading")
                return True
                
            except ImportError:
                print("⚠️ Waitress not available, trying Gunicorn...")
                
                # Try Gunicorn (Unix/Linux only, not in threads)
                try:
                    import gunicorn
                    print("❌ Gunicorn cannot run in Streamlit threads. Use standalone deployment.")
                    print("💡 Click 'Create Standalone API' to generate a separate server file.")
                    production = False
                    
                except ImportError:
                    print("⚠️ Gunicorn not available, falling back to development server...")
                    production = False
        
        if is_streamlit_cloud and production:
            print("⚠️ Production servers not supported on Streamlit Cloud.")
            print("� Streamlit Cloud limitations prevent running production WSGI servers.")
            print("🔧 Use the standalone API file for external deployment.")
            production = False
        
        if not production:
            # Use Flask development server (thread-safe)
            print("⚠️ Using Flask development server")
            
            def run_server():
                flask_app.run(
                    host='0.0.0.0', 
                    port=port, 
                    debug=False, 
                    use_reloader=False, 
                    threaded=True
                )
            
            api_server_thread = threading.Thread(target=run_server, daemon=True)
            api_server_thread.start()
            
            print(f"✅ Development API server started on http://localhost:{port}")
            print(f"📊 Clustering endpoint: http://localhost:{port}/cluster")
            print(f"🔍 Health check: http://localhost:{port}/health")
            
            if is_streamlit_cloud:
                print("🌐 Note: On Streamlit Cloud, API may only be accessible internally")
            else:
                print("⚠️ WARNING: This is a development server. Use standalone API for production.")
            
            return True
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def create_standalone_api_file():
    """Create a standalone API server file for production deployment"""
    standalone_content = '''#!/usr/bin/env python3
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
'''
    
    with open('api_server.py', 'w') as f:
        f.write(standalone_content)
    
    print("✅ Created standalone API server file: api_server.py")
    print("🚀 To run in production:")
    print("   python api_server.py")
    print("   OR")
    print("   gunicorn -w 2 -b 0.0.0.0:5000 api_server:app")

def stop_api_server():
    """Stop the API server"""
    print("API server will stop when the main process exits.")

# Register cleanup
atexit.register(stop_api_server)

# ==================================================
# STREAMLIT WEB INTERFACE
# ==================================================

if STREAMLIT_CLOUD and not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.warning("⚠️ Running on Streamlit Cloud - automatic installation disabled for stability.")

st.set_page_config(
    page_title="Keyword Clustering Tool",
    page_icon="🔍",
    layout="wide"
)

# Add installation status in sidebar
with st.sidebar:
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st.success("✅ All packages installed!")
        st.caption("This app auto-installs dependencies")
    else:
        st.warning("⚠️ AI model installation pending")
        st.caption("Some features may be limited")
    
    # API Server Controls
    st.subheader("🚀 API Server")
    
    # Check if we're on Streamlit Cloud
    is_streamlit_cloud = STREAMLIT_CLOUD_DETECTED
    
    if FLASK_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
        if is_streamlit_cloud:
            st.info("🌐 Streamlit Cloud Environment")
            st.caption("Production servers not supported in Streamlit Cloud threads")
            
            # Only show development mode and standalone options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚙️ Start Dev Server", help="Start development Flask server"):
                    if start_api_server(5000, production=False):
                        st.success("Development API server started!")
                        st.warning("Note: May only be accessible internally on Streamlit Cloud")
            
            with col2:
                if st.button("📄 Create Standalone API", help="Create standalone API file for external deployment"):
                    create_standalone_api_file()
                    st.success("Created api_server.py for external deployment")
                    
                    # Read and provide download
                    try:
                        with open('api_server.py', 'r') as f:
                            api_content = f.read()
                        st.download_button(
                            label="⬇️ Download API Server",
                            data=api_content,
                            file_name='api_server.py',
                            mime='text/x-python'
                        )
                    except:
                        st.error("Could not read generated file")
            
            st.markdown("💡 **For production API**: Download the standalone file and deploy separately")
            
        else:
            # Local environment - show all options
            server_type = st.radio(
                "Server Type:",
                ["🔒 Production (Waitress)", "⚙️ Development (Flask)"],
                help="Production uses Waitress (thread-safe), Development uses Flask"
            )
            
            port = st.number_input("Port:", min_value=3000, max_value=9999, value=5000)
            
            production_mode = server_type.startswith("🔒")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌐 Start API Server", help=f"Start {'production' if production_mode else 'development'} server"):
                    if start_api_server(port, production=production_mode):
                        st.success(f"{'Production' if production_mode else 'Development'} API server started!")
                        st.info(f"API running on http://localhost:{port}")
            
            with col2:
                if st.button("📄 Create Standalone API", help="Create standalone API file for deployment"):
                    create_standalone_api_file()
                    st.success("Created api_server.py")
                    
                    # Read and provide download
                    try:
                        with open('api_server.py', 'r') as f:
                            api_content = f.read()
                        st.download_button(
                            label="⬇️ Download API Server",
                            data=api_content,
                            file_name='api_server.py',
                            mime='text/x-python'
                        )
                    except:
                        st.error("Could not read generated file")
    else:
        st.warning("API server requires Flask and sentence-transformers")
        if not FLASK_AVAILABLE:
            st.caption("Flask not installed")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.caption("sentence-transformers not available")

st.title('🔍 Keyword Clustering Tool')
st.markdown("Cluster similar keywords using AI-powered semantic analysis")

if SENTENCE_TRANSFORMERS_AVAILABLE:
    st.info("🚀 **Self-Sufficient App**: All dependencies are automatically installed!")
else:
    if STREAMLIT_CLOUD:
        st.error("❌ **Streamlit Cloud Limitation**: AI clustering requires sentence-transformers to be pre-installed.")
        st.markdown("""
        ### For Streamlit Cloud Deployment:
        
        Create a `requirements.txt` file with:
        ```
        streamlit
        sentence-transformers
        pandas
        numpy
        requests
        openpyxl
        torch>=1.9.0
        transformers>=4.21.0
        flask
        flask-cors
        ```
        
        Then deploy with both files: `clustering with python.py` and `requirements.txt`
        """)
    else:
        st.warning("⚠️ **Installation Issues**: The AI model couldn't be installed automatically. Basic functionality available.")
        
        # Add a button to retry installation
        if st.button("🔄 Retry AI Model Installation"):
            st.rerun()

# Best default parameters for optimal performance
DEFAULT_THRESHOLD = 0.75  # Good balance between precision and recall for keywords
DEFAULT_MIN_COMMUNITY_SIZE = 2  # Minimum cluster size
DEFAULT_BATCH_SIZE = 32  # Optimal for most hardware

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
threshold = st.sidebar.slider(
    'Similarity Threshold:', 
    min_value=0.0, 
    max_value=1.0, 
    value=DEFAULT_THRESHOLD, 
    step=0.05,
    help="Higher values create more precise clusters (0.75-0.85 recommended for keywords)"
)

min_community_size = st.sidebar.slider(
    'Minimum Cluster Size:', 
    min_value=1, 
    max_value=10, 
    value=DEFAULT_MIN_COMMUNITY_SIZE, 
    step=1,
    help="Minimum number of keywords required to form a cluster"
)

# Optional webhook configuration
st.sidebar.subheader("🔗 Webhook Settings")
webhook_url = st.sidebar.text_input(
    "Webhook URL (Optional):",
    placeholder="https://your-webhook-url.com/endpoint",
    help="Send results to this URL after clustering"
)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Keywords")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📁 File Upload"],
        horizontal=True
    )
    
    keywords_list = []
    
    if input_method == "📝 Text Input":
        keywords_input = st.text_area(
            "Enter keywords (one per line):",
            height=200,
            placeholder="machine learning\nartificial intelligence\ndata science\ndeep learning\nneural networks\ncomputer vision\nnatural language processing"
        )
        
        if keywords_input:
            keywords_list = [k.strip() for k in keywords_input.splitlines() if k.strip()]
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload a file with keywords:",
            type=['csv', 'xlsx', 'txt'],
            help="CSV/Excel: keywords should be in the first column. TXT: one keyword per line."
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.txt'):
                    content = str(uploaded_file.read(), "utf-8")
                    keywords_list = [k.strip() for k in content.splitlines() if k.strip()]
                elif uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file)
                    except ImportError:
                        st.error("Installing pandas for CSV support...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
                        import pandas as pd
                        df = pd.read_csv(uploaded_file)
                    keywords_list = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                elif uploaded_file.name.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(uploaded_file)
                    except ImportError:
                        st.error("Installing openpyxl for Excel support...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
                        df = pd.read_excel(uploaded_file)
                    except Exception:
                        # Fallback: try installing xlrd for older Excel files
                        try:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "xlrd"])
                            df = pd.read_excel(uploaded_file)
                        except:
                            st.error("❌ Could not read Excel file. Please save as CSV instead.")
                            keywords_list = []
                    if 'df' in locals():
                        keywords_list = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                
                st.success(f"✅ Loaded {len(keywords_list)} keywords from file")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

with col2:
    st.subheader("📊 Quick Stats")
    if keywords_list:
        unique_keywords = list(set(keywords_list))
        st.metric("Total Keywords", len(keywords_list))
        st.metric("Unique Keywords", len(unique_keywords))
        st.metric("Duplicates", len(keywords_list) - len(unique_keywords))
        
        # Show sample keywords
        st.subheader("🔍 Sample Keywords")
        sample_size = min(5, len(unique_keywords))
        for i, keyword in enumerate(unique_keywords[:sample_size]):
            st.text(f"• {keyword}")
        if len(unique_keywords) > sample_size:
            st.text(f"... and {len(unique_keywords) - sample_size} more")

# Process button and results
if keywords_list:
    if st.button("🚀 Start Clustering", type="primary", use_container_width=True):
        
        if len(set(keywords_list)) < 2:
            st.warning("⚠️ Please provide at least 2 unique keywords for clustering.")
        else:
            # Perform clustering
            result = perform_clustering(keywords_list, threshold, min_community_size)
            
            if result:
                # Display results
                st.success(f"🎉 Clustering completed! Found {result['total_clusters']} clusters in {result['processing_time']['total_time']} seconds.")
                
                # Results overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Keywords", result['total_keywords'])
                with col2:
                    st.metric("Clusters Found", result['total_clusters'])
                with col3:
                    st.metric("Unclustered", result['unclustered_count'])
                with col4:
                    st.metric("Processing Time", f"{result['processing_time']['total_time']}s")
                
                # Display clusters
                st.subheader("🗂️ Clustering Results")
                
                # Create DataFrame for display
                display_df = pd.DataFrame([
                    {
                        'Cluster ID': cluster['Cluster ID'],
                        'Cluster Name': cluster['Cluster Name'],
                        'Keyword Count': cluster['Keyword Count'],
                        'Keywords': cluster['Keywords']
                    }
                    for cluster in result['clusters']
                ])
                
                st.dataframe(display_df, use_container_width=True)
                
                # Detailed cluster view
                with st.expander("🔍 Detailed Cluster View"):
                    for cluster in result['clusters']:
                        if cluster['Cluster ID'] == 0:
                            st.subheader(f"🔸 {cluster['Cluster Name']} ({cluster['Keyword Count']} keywords)")
                        else:
                            st.subheader(f"📁 Cluster {cluster['Cluster ID']}: {cluster['Cluster Name']} ({cluster['Keyword Count']} keywords)")
                        
                        # Display keywords as tags
                        cols = st.columns(min(len(cluster['Keyword List']), 5))
                        for idx, keyword in enumerate(cluster['Keyword List']):
                            with cols[idx % 5]:
                                st.code(keyword)
                        st.divider()
                
                # Download options
                st.subheader("💾 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = convert_df(display_df)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv,
                        file_name=f'keyword_clusters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                with col2:
                    json_data = json.dumps(result, indent=2)
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json_data,
                        file_name=f'keyword_clusters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json',
                        use_container_width=True
                    )
                
                # Send webhook if configured
                if webhook_url:
                    if st.button("📤 Send to Webhook", use_container_width=True):
                        send_webhook_result(webhook_url, result)

else:
    st.info("👆 Please enter keywords or upload a file to start clustering.")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Clustering:
- **Threshold 0.75-0.85**: Best for most keyword clustering tasks
- **Threshold 0.85-0.95**: Very strict clustering (high precision)
- **Threshold 0.60-0.75**: Loose clustering (high recall)
- **Min Cluster Size 2**: Most sensitive to similarities
- **Keywords work best**: Use single words or short phrases rather than full sentences
""")

st.markdown("""
### 🚀 Self-Sufficient App Features:
- **Auto-Installation**: All required packages are installed automatically (local environments)
- **Single File**: No need for requirements.txt or separate dependencies (for local use)
- **Streamlit Cloud Ready**: Use with requirements.txt for cloud deployment
- **Error Recovery**: Graceful handling when auto-installation isn't possible
- **Built-in API**: Flask API server for n8n and other integrations

### 🌟 Deployment Options:

**For Local Development:**
- Just run this single file - auto-installation handles everything!
- Start the API server from the sidebar for n8n integration

**For Streamlit Cloud:**
- Upload both `clustering with python.py` and `requirements.txt`
- The app will detect cloud environment and skip auto-installation
- All dependencies will be installed via requirements.txt
- **Note**: Production API servers cannot run within Streamlit Cloud due to threading limitations
- Use the standalone API file for external production deployment

**For Production API Deployment:**
- Download the standalone API file using the sidebar button
- Deploy separately on your preferred platform (Heroku, Railway, Google Cloud Run, etc.)
- The standalone file includes production server selection (Waitress → Gunicorn → Flask fallback)

### 🔧 Production Server Priority:
1. **Waitress** (Recommended) - Cross-platform, thread-safe, works everywhere
2. **Gunicorn** - Unix/Linux only, process-based workers
3. **Flask Dev Server** - Fallback only, not for production

### 🔗 API Integration:
**API Endpoints:**
- `POST /cluster` - Cluster keywords
- `GET /health` - Health check
- `GET /` - API information

**Example cURL:**
```bash
curl -X POST http://localhost:5000/cluster \\
  -H "Content-Type: application/json" \\
  -d '{
    "keywords": ["machine learning", "AI", "data science"],
    "threshold": 0.75,
    "min_community_size": 2,
    "webhook_url": "https://your-webhook.com/endpoint"
  }'
```

### ⚠️ Streamlit Cloud Limitations:
- Production WSGI servers (Gunicorn/Waitress) cannot run in Streamlit threads
- Signal handling restrictions prevent proper server initialization
- For production API: use the standalone file on external platforms
- Streamlit Cloud is perfect for the web interface, external deployment for API
""")

# Installation guide for manual deployment
with st.expander("🛠️ Manual Installation Guide"):
    st.markdown("""
    If automatic installation fails, you can manually install dependencies:
    
    ```bash
    # Core dependencies
    pip install streamlit sentence-transformers pandas numpy requests
    
    # API dependencies
    pip install flask flask-cors
    
    # Optional dependencies for file support
    pip install openpyxl xlrd
    
    # PyTorch (CPU version for most deployments)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    
    **For Streamlit Cloud**: Upload both this file and the requirements.txt file for automatic installation!
    """)

# Example JSON for API reference
with st.expander("📋 Example API Payload"):
    example_payload = {
        "keywords": [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural networks",
            "data science"
        ],
        "threshold": 0.75,
        "min_community_size": 2,
        "webhook_url": "https://your-webhook-url.com/endpoint"
    }
    st.code(json.dumps(example_payload, indent=2), language='json')

# App information
st.markdown("---")
st.caption("🔍 Keyword Clustering Tool | Self-Sufficient Single File App | Web Interface + API Server")
