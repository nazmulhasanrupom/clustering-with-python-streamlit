from sentence_transformers import SentenceTransformer, util
import os
import time
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import requests
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Best default parameters for optimal performance
DEFAULT_THRESHOLD = 0.75  # Good balance between precision and recall for keywords
DEFAULT_MIN_COMMUNITY_SIZE = 2  # Minimum cluster size
DEFAULT_BATCH_SIZE = 32  # Optimal for most hardware

# Global model variable to avoid reloading
model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading the sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    return model

def get_median_title_cluster(cluster, corpus_keywords):
    """Get the median length keyword from cluster as representative title"""
    title_lens = [len(corpus_keywords[i]) for i in cluster]
    median_idx = cluster[np.argsort(title_lens)[len(title_lens) // 2]]
    return corpus_keywords[median_idx]

def perform_clustering(keywords, threshold=None, min_community_size=None):
    """Perform clustering on the provided keywords"""
    
    # Use default values if not provided
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if min_community_size is None:
        min_community_size = DEFAULT_MIN_COMMUNITY_SIZE
    
    logger.info(f"Starting clustering with {len(keywords)} keywords")
    logger.info(f"Parameters: threshold={threshold}, min_community_size={min_community_size}")
    
    # Load model
    keyword_model = load_model()
    
    # Clean and deduplicate keywords
    corpus_keywords = list(set([k.strip() for k in keywords if k.strip()]))
    
    if not corpus_keywords:
        raise ValueError("No valid keywords found in the data")
    
    logger.info(f"Encoding {len(corpus_keywords)} unique keywords...")
    start_encoding = time.time()
    corpus_embeddings = keyword_model.encode(
        corpus_keywords, 
        batch_size=DEFAULT_BATCH_SIZE, 
        show_progress_bar=False, 
        convert_to_tensor=True
    )
    encoding_time = time.time() - start_encoding
    logger.info(f"Encoding completed in {encoding_time:.2f} seconds")
    
    # Perform clustering
    logger.info("Starting clustering...")
    start_clustering = time.time()
    clusters = util.community_detection(
        corpus_embeddings, 
        min_community_size=min_community_size, 
        threshold=threshold
    )
    clustering_time = time.time() - start_clustering
    logger.info(f"Clustering completed in {clustering_time:.2f} seconds")
    
    # Prepare results
    clusters_data = []
    clustered_keywords = set()
    
    for i, cluster in enumerate(clusters):
        cluster_name = get_median_title_cluster(cluster, corpus_keywords)
        cluster_keywords = [corpus_keywords[keyword_id] for keyword_id in cluster]
        
        clusters_data.append({
            'cluster_id': i + 1,
            'cluster_name': cluster_name,
            'keywords': cluster_keywords,
            'keyword_count': len(cluster_keywords)
        })
        
        # Track clustered keywords
        clustered_keywords.update(cluster_keywords)
    
    # Handle unclustered keywords
    unclustered_keywords = [k for k in corpus_keywords if k not in clustered_keywords]
    if unclustered_keywords:
        clusters_data.append({
            'cluster_id': 0,
            'cluster_name': 'Unclustered',
            'keywords': unclustered_keywords,
            'keyword_count': len(unclustered_keywords)
        })
    
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
            'threshold': threshold,
            'min_community_size': min_community_size
        },
        'clusters': clusters_data,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Clustering completed successfully. Found {len(clusters)} clusters.")
    return result

def send_webhook_result(webhook_url, result_data):
    """Send the clustering result to the specified webhook URL"""
    try:
        logger.info(f"Sending result to webhook: {webhook_url}")
        response = requests.post(
            webhook_url, 
            json=result_data, 
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Webhook sent successfully. Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send webhook: {str(e)}")
        return False

@app.route('/cluster', methods=['POST'])
def cluster_endpoint():
    """Main endpoint for clustering keywords"""
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract keywords from request (support both 'keywords' and 'sentences' for backward compatibility)
        keywords = data.get('keywords', data.get('sentences', []))
        if not keywords:
            return jsonify({'error': 'No keywords provided'}), 400
        
        if not isinstance(keywords, list):
            return jsonify({'error': 'Keywords must be provided as a list'}), 400
        
        # Extract optional parameters
        threshold = data.get('threshold', DEFAULT_THRESHOLD)
        min_community_size = data.get('min_community_size', DEFAULT_MIN_COMMUNITY_SIZE)
        webhook_url = data.get('webhook_url')
        
        # Validate parameters
        if not (0.0 <= threshold <= 1.0):
            return jsonify({'error': 'Threshold must be between 0.0 and 1.0'}), 400
        
        if not (1 <= min_community_size <= 20):
            return jsonify({'error': 'Min community size must be between 1 and 20'}), 400
        
        # Perform clustering
        result = perform_clustering(keywords, threshold, min_community_size)
        
        # Send to webhook if provided
        if webhook_url:
            webhook_success = send_webhook_result(webhook_url, result)
            result['webhook_sent'] = webhook_success
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'service': 'Keyword Clustering API',
        'version': '1.0',
        'endpoints': {
            'POST /cluster': 'Main clustering endpoint',
            'GET /health': 'Health check',
            'GET /': 'This documentation'
        },
        'example_request': {
            'url': '/cluster',
            'method': 'POST',
            'body': {
                'keywords': [
                    'machine learning',
                    'artificial intelligence', 
                    'deep learning',
                    'neural networks',
                    'data science',
                    'computer vision',
                    'natural language processing',
                    'python programming',
                    'web development'
                ],
                'threshold': 0.75,
                'min_community_size': 2,
                'webhook_url': 'https://your-webhook-url.com/endpoint'
            }
        }
    }), 200

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
