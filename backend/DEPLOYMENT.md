# Phase 3 Cloud Deployment Guide

This document details the production deployment specifications, commands, environment variables, and health probes for LocalGPT Phase 3.

---

## 1. Production Architecture Overview

The system consists of three deployment tiers:

1. **Frontend (Next.js 16 + React 19 + TypeScript)**:
   - Client-side static assets with server rendering.
   - Communicates with FastAPI via REST and Server-Sent Events (SSE).
2. **Backend API (FastAPI + Async Python 3.10+)**:
   - Asynchronous ASGI service handling auth, document ingestion, vector retrieval, and LLM streaming.
   - Runs with Uvicorn multi-worker process manager.
3. **Data Layer**:
   - **PostgreSQL**: Relational persistence for users, conversations, messages, and document metadata.
   - **Persistent Volume Mounts**: Local/NFS/EBS block storage for isolated user documents and FAISS indices.
   - **Hosted LLM Endpoint**: OpenAI-compatible external API (e.g. OpenAI, Groq, Together, vLLM).

---

## 2. Production Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Default / Example | Purpose |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | Yes | `production` | Declares operational mode |
| `DEBUG` | Yes | `False` | Disables debug stacktraces |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host:5432/db` | Async PostgreSQL connection string |
| `SECRET_KEY` | Yes | `<64-hex-random-string>` | JWT signature key |
| `CORS_ORIGINS` | Yes | `https://app.yourdomain.com` | Allowed frontend origins (comma-separated or JSON) |
| `LLM_PROVIDER` | Yes | `hosted_endpoint` | Primary completion provider |
| `HOSTED_LLM_API_KEY` | Yes | `sk-...` | Hosted LLM provider API token |
| `HOSTED_LLM_API_BASE` | No | `https://api.openai.com/v1` | Base URL for OpenAI-compatible endpoint |
| `HOSTED_LLM_MODEL` | No | `gpt-4o-mini` | Target model name |
| `UPLOAD_STORAGE_DIR` | Yes | `/var/lib/localgpt/documents` | Persistent volume path for documents |
| `FAISS_INDEX_DIR` | Yes | `/var/lib/localgpt/vector_store` | Persistent volume path for vector store |
| `GOOGLE_CLIENT_ID` | Optional | `<id>.apps.googleusercontent.com` | Google OAuth 2.0 Client ID |
| `GOOGLE_CLIENT_SECRET` | Optional | `<secret>` | Google OAuth 2.0 Client Secret |

### Frontend (`frontend/.env.production`)

| Variable | Required | Default / Example | Purpose |
| :--- | :--- | :--- | :--- |
| `NEXT_PUBLIC_API_BASE_URL` | Yes | `https://api.yourdomain.com/api/v1` | Public backend API URL |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID`| Optional | `<id>.apps.googleusercontent.com` | Client-side Google Sign-In button |

---

## 3. Deployment Commands

### A. Database Migrations (Run once before starting backend)
```bash
cd backend
alembic upgrade head
```

### B. Backend Production Startup Command
Run with Uvicorn ASGI server with production worker pooling:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers --forwarded-allow-ips='*'
```
*Note: In Windows environments, worker pooling defaults to 1 worker per process or system-managed task runner.*

### C. Frontend Production Build & Start Commands
```bash
cd frontend
npm ci
npm run build
npm start -- -p 3000
```

---

## 4. Health & Readiness Probes (For Load Balancers / Orchestrators)

| Probe Type | URL | Expected Code | Behavior |
| :--- | :--- | :--- | :--- |
| **Liveness Probe** | `GET /health` | `200 OK` | Confirms ASGI worker is active and accepting requests |
| **Readiness Probe** | `GET /ready` | `200 OK` / `503 Service Unavailable` | Checks PostgreSQL connection; returns 503 if DB is unreachable |
| **API Diagnostic** | `GET /api/v1/health` | `200 OK` | Detailed JSON breakdown of database, RAG, and LLM subsystem status |

---

## 5. Persistent Storage Mounts

Ensure cloud containers mount persistent storage volumes at:
- `UPLOAD_STORAGE_DIR`: Retains original uploaded files partitioned by user UUID.
- `FAISS_INDEX_DIR`: Retains binary FAISS vector indices and chunk metadata per user UUID.
