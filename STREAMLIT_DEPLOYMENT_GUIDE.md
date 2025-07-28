# Streamlit Cloud Deployment Guide

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository
1. Upload these files to your GitHub repository:
   - `clustering with python.py` (main app)
   - `requirements.txt` (dependencies)
   - `packages.txt` (system packages)
   - `.streamlit/config.toml` (configuration)

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set main file path: `clustering with python.py`
5. Click "Deploy!"

### Step 3: Wait for Deployment
- Initial deployment may take 5-10 minutes
- Model loading will add another 1-2 minutes on first use
- Subsequent uses will be much faster due to caching

## 📋 Required Files Structure

```
your-repo/
├── clustering with python.py          # Main Streamlit app
├── requirements.txt                   # Python dependencies
├── packages.txt                       # System packages (optional)
├── .streamlit/
│   └── config.toml                   # Streamlit configuration
└── README.md                         # Documentation (optional)
```

## 🔧 Troubleshooting

### Memory Issues
If you encounter memory errors on Streamlit Cloud:
1. The free tier has limited memory (~1GB)
2. The sentence-transformer model needs ~400MB
3. Consider upgrading to Streamlit Cloud Pro for more resources

### Model Loading Errors
- First model load takes 1-2 minutes - be patient
- If it fails, try refreshing the page
- Check that all dependencies are in requirements.txt

### Import Errors
- Make sure `requirements.txt` includes all packages
- Version conflicts can cause issues - stick to the provided versions
- Streamlit Cloud uses Python 3.9+ by default

## 📝 requirements.txt Content

```
sentence-transformers>=2.2.2
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
torch>=1.9.0
transformers>=4.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
huggingface-hub>=0.10.0
tokenizers>=0.13.0
openpyxl>=3.0.0
```

## 🌐 Custom Domain (Optional)

After deployment, you can:
1. Use the default `.streamlit.app` URL
2. Set up a custom domain in Streamlit Cloud settings
3. Share your app with the generated URL

## 💡 Performance Tips

1. **First Load**: Will be slow (1-2 minutes) due to model download
2. **Subsequent Loads**: Much faster due to Streamlit caching
3. **Memory Usage**: Monitor in Streamlit Cloud dashboard
4. **Timeout**: Large datasets may timeout - consider chunking

## 🔒 Security Considerations

- Don't include API keys or sensitive data in your repository
- Use Streamlit secrets for sensitive configuration
- The webhook feature should point to HTTPS URLs only

## 📊 Usage Limits

Streamlit Cloud Free Tier:
- 1 GB RAM
- 1 CPU core
- 10 GB bandwidth/month
- 3 active apps

For production use with heavy traffic, consider upgrading to Streamlit Cloud Pro.
