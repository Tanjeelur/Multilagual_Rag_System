API Documentation
Overview
The Multilingual RAG API provides a RESTful interface for querying a Retrieval-Augmented Generation system supporting English and Bengali queries.
Endpoint: POST /query
Request

Method: POST
URL: /query
Body:{
  "query": "string",
  "language": "bn" | "en"
}


query: The query string (e.g., "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?").
language: Output language (bn for Bengali, en for English, default: bn).



Response

Status: 200 OK
Body:{
  "answer": "string",
  "confidence": float
}


answer: The generated answer.
confidence: Cosine similarity score (0.0 to 1.0).



Example
Request:
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "language": "bn"}'

Response:
{
  "answer": "শুম্ভুনাথ",
  "confidence": 0.85
}

Error Responses

400 Bad Request: Invalid input (e.g., missing query).
500 Internal Server Error: Server-side issues (e.g., database failure).

Accessing Docs
Interactive API documentation is available at http://localhost:8000/docs when the server is running.