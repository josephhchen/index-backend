# Vector Search API

Go backend service for semantic product search using Weaviate vector database and OpenAI embeddings.

## Features

- **Vector Search**: Semantic product search using OpenAI embeddings
- **Product Recommendations**: Similar product suggestions based on vector similarity
- **RESTful API**: Clean API endpoints for frontend integration
- **Auto-vectorization**: Automatic product data indexing with Weaviate

## Tech Stack

- **Go 1.21+**: High-performance backend
- **Weaviate**: Vector database for semantic search
- **OpenAI API**: Text embeddings (text-embedding-ada-002)
- **Gin**: Web framework for REST API

## Quick Start

### Prerequisites
- Go 1.21 or higher
- Docker and Docker Compose
- OpenAI API key

### 1. Environment Setup
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Start Weaviate
```bash
docker-compose up -d
```

### 3. Run Backend
```bash
go mod tidy
go run main.go
```

### Get Recommendations
```http
GET /recommendations?product=iPhone%2015%20Pro&limit=5
```

### Health Check
```http
GET /health
```

## Configuration

### Environment Variables
- `WEAVIATE_HOST`: Weaviate server address (default: localhost:8080)
- `WEAVIATE_API_KEY`: Weaviate API key (optional for local)
- `OPENAI_API_KEY`: OpenAI API key (required)
- `PORT`: Server port (default: 8000)

## License

Educational use only.# index-backend
# index-backend
