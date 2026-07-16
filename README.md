# =============================================================================
#  NutriAI – AI Powered Nutrition Agent
#  README.md — Complete Setup & Deployment Guide
#  Powered by IBM watsonx.ai Granite Models on IBM Cloud
# =============================================================================

# 🥗 NutriAI – AI Powered Nutrition Agent

> A multi-agent AI nutrition assistant powered by **IBM watsonx.ai Granite Models**, **IBM Langflow**, and **IBM Cloud infrastructure**.

[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0062ff)](https://www.ibm.com/watsonx)
[![IBM Cloud](https://img.shields.io/badge/IBM-Cloud-0062ff)](https://cloud.ibm.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_18-61dafb)](https://reactjs.org)

---

## 📋 Table of Contents
1. [Project Overview](#overview)
2. [Architecture](#architecture)
3. [Four AI Agents](#agents)
4. [IBM Cloud Services Used](#ibm-services)
5. [Prerequisites](#prerequisites)
6. [Local Development Setup](#local-setup)
7. [RAG Data Ingestion](#rag-ingestion)
8. [IBM Cloud Deployment](#ibm-deployment)
9. [API Reference](#api-reference)
10. [Folder Structure](#folder-structure)
11. [Environment Variables](#env-vars)

---

## 🌟 Project Overview <a name="overview"></a>

NutriAI is a **multi-agent AI nutrition assistant** built for IBM hackathons and IBM SkillsBuild showcases. It demonstrates:

- **IBM watsonx.ai Granite 13B** as the primary LLM for all four agents
- **Retrieval-Augmented Generation (RAG)** over USDA, WHO, and IFCT nutrition data
- **IBM Langflow** for visual multi-agent workflow orchestration
- **IBM Cloud** (Code Engine, Db2, COS, Redis, App ID) as the full deployment platform

---

## 🏗 Architecture <a name="architecture"></a>

```
User (React/PWA)
    │
    ▼
IBM API Gateway (Auth, Rate Limiting, SSL)
    │
    ▼
FastAPI Backend (IBM Code Engine)
    │
    ▼
IBM Langflow Orchestrator
    │
    ├─► Agent 1 – Nutrition Knowledge  (Granite 13B + Milvus RAG)
    ├─► Agent 2 – Diet Recommendation  (Granite 13B + User Profile)
    ├─► Agent 3 – Health Advisory      (Granite 13B + Clinical RAG)
    └─► Agent 4 – Food Log & Feedback  (Granite 3B/13B + STT/Vision)
              │
              ▼
    IBM Cloud Data Layer
    ├── Milvus Vector DB  (USDA/WHO/IFCT embeddings)
    ├── IBM Cloud Db2     (Users, Food Logs, Meal Plans)
    ├── IBM Cloud COS     (Documents, Food Images)
    └── IBM Redis         (Session/Prompt Cache)
```

---

## 🤖 Four AI Agents <a name="agents"></a>

| Agent | Model | Technique | Endpoint |
|-------|-------|-----------|----------|
| **1 – Nutrition Knowledge** | `granite-13b-instruct-v2` | RAG (USDA/WHO/IFCT) | `POST /api/v1/nutrition/query` |
| **2 – Diet Recommendation** | `granite-13b-instruct-v2` | Profile-aware prompting | `POST /api/v1/diet/plan` |
| **3 – Health Advisory** | `granite-13b-instruct-v2` | Clinical guideline RAG | `POST /api/v1/health/advisory` |
| **4 – Food Log & Feedback** | `granite-3b` + `granite-13b` | Text/Image/Voice | `POST /api/v1/foodlog/text` |

---

## ☁️ IBM Cloud Services <a name="ibm-services"></a>

| Service | Purpose |
|---------|---------|
| **IBM watsonx.ai** | Granite 13B (agents), Granite 3B (classifier, fast analysis), Slate-30m (embeddings) |
| **IBM Langflow** | Visual multi-agent orchestration |
| **IBM Code Engine** | Serverless container hosting for FastAPI |
| **IBM Cloud Object Storage** | USDA/WHO/IFCT documents, food images |
| **IBM Cloud Db2** | User profiles, food logs, meal plans |
| **IBM Databases for Redis** | Prompt & session caching |
| **IBM Watson Speech-to-Text** | Voice meal logging (Agent 4) |
| **IBM App ID** | OAuth2 / JWT authentication |
| **IBM API Gateway** | Rate limiting, SSL termination |
| **IBM Secrets Manager** | Secure credential storage |
| **IBM Container Registry** | Docker image storage |
| **IBM Cloud Monitoring** | Metrics & alerts |

---

## 📦 Prerequisites <a name="prerequisites"></a>

- Python 3.11+
- Node.js 20+
- Docker + Docker Compose
- IBM Cloud account with watsonx.ai access
- IBM Cloud CLI (`ibmcloud`)

Install IBM Cloud CLI plugins:
```bash
ibmcloud plugin install code-engine
ibmcloud plugin install container-registry
ibmcloud plugin install secrets-manager
```

---

## 🚀 Local Development Setup <a name="local-setup"></a>

### 1. Clone & setup environment

```bash
git clone https://github.com/your-org/nutriai.git
cd nutriai

# Backend
cp backend/.env.example backend/.env
# Edit backend/.env with your IBM credentials

# Frontend
cd frontend
cp .env.example .env.local
# Set REACT_APP_API_URL=http://localhost:8000
```

### 2. Start with Docker Compose (recommended)

```bash
# From nutriai/ root
docker-compose up --build
```

Services started:
- **Backend API**:  http://localhost:8000
- **Frontend**:     http://localhost:3000
- **Milvus**:       localhost:19530
- **Redis**:        localhost:6379

### 3. Manual backend setup

```bash
cd backend
pip install -r requirements.txt

# Activate env vars
export WATSONX_API_KEY=your_key
export WATSONX_PROJECT_ID=your_project_id

# Start FastAPI
uvicorn main:app --reload --port 8000
```

### 4. Manual frontend setup

```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

### 5. API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI with all endpoints.

---

## 📚 RAG Data Ingestion <a name="rag-ingestion"></a>

Before using the Nutrition Knowledge and Health Advisory agents, you must ingest data into Milvus.

### Step 1: Upload data to IBM COS

```bash
# Download USDA FoodData Central (SR Legacy)
# https://fdc.nal.usda.gov/download-datasets.html
# Rename to usda_foods.json and upload:

ibmcloud cos upload \
  --bucket nutriai-rag-data \
  --key usda_foods.json \
  --file /path/to/usda_foods.json

# Upload IFCT 2017 CSV
ibmcloud cos upload --bucket nutriai-rag-data --key ifct_2017.csv --file ifct_2017.csv

# Upload WHO/AHA/ADA PDFs
ibmcloud cos upload --bucket nutriai-rag-data --key who_healthy_diet.pdf --file who_diet.pdf
```

### Step 2: Run ingestion scripts

```bash
cd backend

# Ingest USDA (largest – ~600K foods)
python scripts/ingest_usda.py

# Ingest WHO + AHA/ADA guidelines
python scripts/ingest_who.py

# Ingest IFCT 2017 Indian foods
python scripts/ingest_ifct.py
```

Each script embeds chunks using **`ibm/slate-30m-english-rtrvr`** and stores vectors in Milvus.

---

## 🌐 IBM Cloud Deployment <a name="ibm-deployment"></a>

### Step 1: Login and target region

```bash
ibmcloud login --apikey $IBM_CLOUD_API_KEY -r us-south
ibmcloud target -g Default
```

### Step 2: Build and push Docker image

```bash
# Tag and push backend image
ibmcloud cr login
docker build -t us.icr.io/nutriai/backend:latest ./backend
docker push us.icr.io/nutriai/backend:latest

# Tag and push frontend image
docker build -t us.icr.io/nutriai/frontend:latest ./frontend
docker push us.icr.io/nutriai/frontend:latest
```

### Step 3: Create IBM Secrets Manager secret

```bash
ibmcloud secrets-manager secret-create \
  --secret-type kv \
  --secret-name nutriai-env \
  --secret-payload '{"WATSONX_API_KEY":"...","WATSONX_PROJECT_ID":"...","DB2_DSN":"..."}'
```

### Step 4: Deploy FastAPI to IBM Code Engine

```bash
ibmcloud ce project create --name nutriai-project

ibmcloud ce application create \
  --name nutriai-backend \
  --image us.icr.io/nutriai/backend:latest \
  --port 8000 \
  --min-scale 1 \
  --max-scale 10 \
  --env-from-secret nutriai-env \
  --registry-secret icr-secret

# Get the deployment URL
ibmcloud ce application get --name nutriai-backend | grep URL
```

### Step 5: Deploy Frontend to IBM COS Static Site

```bash
cd frontend
REACT_APP_API_URL=https://your-backend-url.ibm.com npm run build

# Upload to COS static website bucket
ibmcloud cos bucket-website-put --bucket nutriai-frontend --website-configuration '{"IndexDocument":{"Suffix":"index.html"}}'

aws s3 sync build/ s3://nutriai-frontend \
  --endpoint-url https://s3.us-south.cloud-object-storage.appdomain.cloud
```

### Step 6: Import Langflow workflow

1. Open IBM Langflow in your watsonx.ai instance
2. Import `langflow/nutriai_workflow.json`
3. Set Granite model credentials
4. Enable webhook endpoint
5. Update `LANGFLOW_URL` in backend env

---

## 📡 API Reference <a name="api-reference"></a>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/api/v1/users/register` | Create user profile |
| `GET`  | `/api/v1/users/{id}` | Get user profile |
| `PUT`  | `/api/v1/users/{id}` | Update user profile |
| `GET`  | `/api/v1/users/{id}/dashboard` | Get dashboard stats |
| `POST` | `/api/v1/nutrition/query` | Agent 1: Nutrition knowledge (RAG) |
| `POST` | `/api/v1/diet/plan` | Agent 2: Generate meal plan |
| `GET`  | `/api/v1/diet/plan/{id}` | Get saved meal plan |
| `POST` | `/api/v1/health/advisory` | Agent 3: Health advisory |
| `POST` | `/api/v1/foodlog/text` | Agent 4: Log meal via text |
| `POST` | `/api/v1/foodlog/image` | Agent 4: Log meal via image |
| `POST` | `/api/v1/foodlog/voice` | Agent 4: Log meal via voice |
| `GET`  | `/api/v1/foodlog/{id}` | Get food log history |

Full interactive docs: `http://localhost:8000/docs`

---

## 📁 Folder Structure <a name="folder-structure"></a>

```
nutriai/
├── backend/
│   ├── main.py                   # FastAPI entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   ├── core/
│   │   ├── watsonx_client.py     # IBM watsonx.ai Granite wrapper
│   │   ├── rag_pipeline.py       # Milvus RAG (embed + retrieve)
│   │   ├── db_client.py          # IBM Cloud Db2
│   │   ├── cos_client.py         # IBM Cloud Object Storage
│   │   └── redis_client.py       # IBM Databases for Redis
│   ├── agents/
│   │   ├── orchestrator.py       # Intent routing orchestrator
│   │   ├── nutrition_agent.py    # Agent 1
│   │   ├── diet_agent.py         # Agent 2
│   │   ├── health_agent.py       # Agent 3
│   │   └── foodlog_agent.py      # Agent 4
│   ├── routes/
│   │   ├── nutrition.py
│   │   ├── diet.py
│   │   ├── health.py
│   │   ├── foodlog.py
│   │   └── users.py
│   ├── models/
│   │   └── schemas.py            # Pydantic v2 schemas
│   └── scripts/
│       ├── ingest_usda.py        # USDA RAG ingestion
│       ├── ingest_who.py         # WHO/AHA/ADA ingestion
│       └── ingest_ifct.py        # IFCT 2017 ingestion
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   └── src/
│       ├── App.jsx               # Root + sidebar layout
│       ├── App.css               # IBM Carbon-inspired styles
│       ├── pages/
│       │   ├── Dashboard.jsx     # Stats + chart
│       │   ├── NutritionChat.jsx # Agent 1 UI
│       │   ├── DietPlanner.jsx   # Agent 2 UI
│       │   ├── HealthAdvisor.jsx # Agent 3 UI
│       │   ├── FoodLogger.jsx    # Agent 4 UI (text/image/voice)
│       │   └── Profile.jsx       # User profile management
│       └── api/
│           └── nutriaiApi.js     # Axios API client
├── langflow/
│   └── nutriai_workflow.json     # IBM Langflow multi-agent workflow
├── docker-compose.yml
└── README.md
```

---

## 🔐 Environment Variables <a name="env-vars"></a>

See [`backend/.env.example`](backend/.env.example) for the complete list.

**Required minimum to run:**
```
WATSONX_API_KEY=
WATSONX_PROJECT_ID=
WATSONX_URL=https://us-south.ml.cloud.ibm.com
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## ⚕️ Medical Disclaimer

> NutriAI is for **educational purposes only**. All dietary and health information generated by AI agents does not constitute medical advice. Users with health conditions must consult a qualified healthcare professional before modifying their diet or lifestyle.

---

## 🏆 Built For

- IBM hackathons
- IBM SkillsBuild AI showcases
- College AI project demonstrations
- IBM watsonx.ai partner demos

---

*Powered by IBM watsonx.ai Granite Models · Built on IBM Cloud*
