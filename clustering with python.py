import subprocess
import sys
import importlib

def install_and_import(package_name, import_name=None):
    """Install and import a package if it's not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            importlib.import_module(import_name)
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            return False

# Auto-install required packages
required_packages = [
    ("streamlit", "streamlit"),
    ("sentence-transformers", "sentence_transformers"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("openpyxl", "openpyxl")
]

print("Checking and installing required packages...")
for package, import_name in required_packages:
    if not install_and_import(package, import_name):
        print(f"ERROR: Could not install {package}. Please install manually with: pip install {package}")
        sys.exit(1)

# Now import everything
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os
import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

st.set_page_config(
    page_title="Keyword Clustering Tool",
    page_icon="🔍",
    layout="wide"
)

# Add installation status in sidebar
with st.sidebar:
    st.success("✅ All packages installed!")
    st.caption("This app auto-installs dependencies")

st.title('🔍 Keyword Clustering Tool')
st.markdown("Cluster similar keywords using AI-powered semantic analysis")
st.info("🚀 **Self-Sufficient App**: All dependencies are automatically installed!")

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

# Model caching with better error handling and auto-install fallback
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with st.spinner("🤖 Loading AI model... This may take up to 2 minutes on first run."):
            # Try to import sentence_transformers again in case it was just installed
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                st.error("Installing sentence-transformers...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.markdown("""
        ### Troubleshooting
        
        If you're getting errors:
        1. The model requires ~400MB of RAM
        2. First load may take 1-2 minutes
        3. Ensure you have a stable internet connection for model download
        4. Try refreshing the page if installation fails
        
        For local development, you can manually install with:
        ```bash
        pip install sentence-transformers torch transformers
        ```
        """)
        # Try manual installation as fallback
        try:
            st.info("Attempting manual installation of dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            st.success("✅ Dependencies installed! Please refresh the page.")
            st.stop()
        except:
            st.error("❌ Automatic installation failed. Please install manually.")
            st.stop()
        return None

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def get_median_title_cluster(cluster, corpus_keywords):
    """Get the median length keyword from cluster as representative title"""
    title_lens = [len(corpus_keywords[i]) for i in cluster]
    median_idx = cluster[np.argsort(title_lens)[len(title_lens) // 2]]
    return corpus_keywords[median_idx]

def perform_clustering(keywords, threshold_val, min_community_size_val):
    """Perform clustering on the provided keywords"""
    
    # Load model
    model = load_model()
    
    # Clean and deduplicate keywords
    corpus_keywords = list(set([k.strip() for k in keywords if k.strip()]))
    
    if not corpus_keywords:
        st.error("❌ No valid keywords found in the data.")
        return None
    
    if len(corpus_keywords) < min_community_size_val:
        st.warning(f"⚠️ Only {len(corpus_keywords)} unique keywords found. Consider reducing minimum cluster size.")
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Encoding phase
    status_text.text(f"🔤 Encoding {len(corpus_keywords)} unique keywords...")
    progress_bar.progress(25)
    
    start_encoding = time.time()
    corpus_embeddings = model.encode(
        corpus_keywords, 
        batch_size=DEFAULT_BATCH_SIZE, 
        show_progress_bar=False, 
        convert_to_tensor=True
    )
    encoding_time = time.time() - start_encoding
    
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
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return result

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
### � Self-Sufficient App Features:
- **Auto-Installation**: All required packages are installed automatically
- **Single File**: No need for requirements.txt or separate dependencies
- **Streamlit Cloud Ready**: Deploy directly without configuration files
- **Error Recovery**: Automatic fallback installation methods
""")

st.markdown("""
### �🔗 API Integration:
For n8n or other integrations, you can send POST requests to this Streamlit app using the webhook feature.
""")

# Installation guide for manual deployment
with st.expander("🛠️ Manual Installation Guide"):
    st.markdown("""
    If automatic installation fails, you can manually install dependencies:
    
    ```bash
    # Core dependencies
    pip install streamlit sentence-transformers pandas numpy requests
    
    # Optional dependencies for file support
    pip install openpyxl xlrd
    
    # PyTorch (CPU version for most deployments)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    
    **For Streamlit Cloud**: Just upload this single file - no requirements.txt needed!
    """)

# Example JSON for API reference
with st.expander("📋 Example Webhook Payload"):
    example_payload = {
        "total_keywords": 10,
        "total_clusters": 3,
        "clusters": [
            {
                "cluster_id": 1,
                "cluster_name": "machine learning",
                "keywords": ["machine learning", "deep learning", "neural networks"],
                "keyword_count": 3
            }
        ],
        "timestamp": "2025-07-28T10:30:45.123456"
    }
    st.code(json.dumps(example_payload, indent=2), language='json')

# App information
st.markdown("---")
st.caption("🔍 Keyword Clustering Tool | Self-Sufficient Single File App | Auto-installs all dependencies")
