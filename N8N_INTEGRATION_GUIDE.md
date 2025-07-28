# Keyword Clustering API - n8n Integration Guide

## Overview
This Flask API provides keyword clustering functionality using sentence transformers. It can receive POST requests from n8n workflows and optionally send results to a webhook.

## API Endpoints

### 1. POST /cluster
Main endpoint for clustering keywords.

**URL:** `http://localhost:5000/cluster`

**Method:** `POST`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "keywords": [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural networks",
    "data science",
    "computer vision",
    "natural language processing"
  ],
  "threshold": 0.75,
  "min_community_size": 2,
  "webhook_url": "https://your-webhook-url.com/endpoint"
}
```

**Request Parameters:**
- `keywords` (required): Array of keywords/phrases to cluster
- `threshold` (optional): Similarity threshold (0.0-1.0, default: 0.75)
- `min_community_size` (optional): Minimum cluster size (1-20, default: 2)
- `webhook_url` (optional): URL to send results to after processing

**Response Example:**
```json
{
  "total_keywords": 7,
  "total_clusters": 2,
  "unclustered_count": 1,
  "processing_time": {
    "encoding_time": 0.85,
    "clustering_time": 0.02,
    "total_time": 0.87
  },
  "parameters": {
    "threshold": 0.75,
    "min_community_size": 2
  },
  "clusters": [
    {
      "cluster_id": 1,
      "cluster_name": "machine learning",
      "keywords": [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural networks"
      ],
      "keyword_count": 4
    },
    {
      "cluster_id": 2,
      "cluster_name": "data science",
      "keywords": [
        "data science",
        "computer vision"
      ],
      "keyword_count": 2
    },
    {
      "cluster_id": 0,
      "cluster_name": "Unclustered",
      "keywords": [
        "natural language processing"
      ],
      "keyword_count": 1
    }
  ],
  "timestamp": "2025-07-28T10:30:45.123456",
  "webhook_sent": true
}
```

### 2. GET /health
Health check endpoint.

**URL:** `http://localhost:5000/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-28T10:30:45.123456",
  "model_loaded": true
}
```

### 3. GET /
API documentation endpoint.

## n8n Workflow Setup

### Step 1: HTTP Request Node
1. Add an **HTTP Request** node in your n8n workflow
2. Set **Method** to `POST`
3. Set **URL** to `http://localhost:5000/cluster` (or your server URL)
4. In **Headers**, add:
   - Key: `Content-Type`
   - Value: `application/json`

### Step 2: Request Body
In the **Body** section of the HTTP Request node, select **JSON** and use this structure:

```json
{
  "sentences": {{ $json.sentences }},
  "threshold": 0.75,
  "min_community_size": 2,
  "webhook_url": "{{ $node.Webhook.json.webhook_url }}"
}
```

### Step 3: Webhook (Optional)
If you want results sent to a webhook:
1. Add a **Webhook** node to receive the results
2. Copy the webhook URL
3. Include it in the `webhook_url` field of your request

## Best Practice Parameters

### Threshold Values:
- **0.85-0.95**: Very strict clustering (high precision, low recall)
- **0.75-0.85**: Balanced clustering (recommended for most use cases)
- **0.60-0.75**: Loose clustering (high recall, lower precision)
- **Below 0.60**: Very loose clustering (may group unrelated sentences)

### Min Community Size:
- **2**: Most sensitive (recommended default)
- **3-5**: Medium sensitivity
- **6+**: Only large, well-defined clusters

## Example n8n JSON Body Templates

### Basic Clustering:
```json
{
  "sentences": [
    "{{ $json.question1 }}",
    "{{ $json.question2 }}",
    "{{ $json.question3 }}"
  ]
}
```

### Advanced Clustering with Custom Parameters:
```json
{
  "sentences": {{ $json.questions_array }},
  "threshold": 0.8,
  "min_community_size": 3,
  "webhook_url": "https://hooks.zapier.com/hooks/catch/123456/abcdef/"
}
```

### Processing CSV Data:
```json
{
  "sentences": {{ $json.csv_data.map(row => row.question) }},
  "threshold": 0.75,
  "min_community_size": 2
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- **200**: Success
- **400**: Bad request (invalid parameters or missing data)
- **500**: Internal server error

Example error response:
```json
{
  "error": "No sentences provided"
}
```

## Performance Notes

- **First Request**: Takes longer due to model loading (~10-30 seconds)
- **Subsequent Requests**: Much faster (~1-5 seconds depending on sentence count)
- **Optimal Batch Size**: 10-500 sentences per request
- **Large Datasets**: Consider splitting into smaller batches

## Deployment

### Local Development:
```bash
python "06_Question_Clustering_Tool(1).py"
```

### Production:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 "06_Question_Clustering_Tool(1):app"
```

### Docker:
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "06_Question_Clustering_Tool(1).py"]
```
