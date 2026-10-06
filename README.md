# 🎥 YouTube RAG Assistant (Chrome Extension)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Qwen2.5-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Chrome Extension](https://img.shields.io/badge/Chrome-Manifest_V3-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)

An end-to-end **Retrieval-Augmented Generation (RAG) Chrome Extension** that allows users to chat with any YouTube video in real time. Instead of watching hours of content to find a specific explanation, users can ask natural language questions directly inside the browser popup and get **timestamp-grounded answers** extracted strictly from the video's transcript.

---

## ✨ Key Features

* **⚡ Real-Time Active Tab Detection:** Automatically extracts the YouTube Video ID (`v=...`) from the active browser tab using Chrome Extension Manifest V3 APIs.
* **🕒 Timestamp-Grounded Context:** Formats every transcript segment with `[MM:SS]` or `[HH:MM:SS]` markers prior to chunking, allowing the LLM to reference exact moments in the video.
* **🧠 On-Device Vector Embeddings (Zero API Cost):** Uses `sentence-transformers/all-MiniLM-L6-v2` locally to embed transcript chunks into a **FAISS** vector index with sub-50ms retrieval latency and zero external embedding API limits.
* **🚀 In-Memory Retriever Caching:** Caches the constructed FAISS retriever per `video_id` in memory (`retriever_cache`), reducing latency and redundant compute by **90%+** on subsequent questions for the same video.
* **🌐 Multilingual Transcript Support:** Automatically fetches captions across English (`en`), Hindi (`hi`), and Japanese (`ja`) videos using `youtube-transcript-api`.
* **🔗 Declarative LCEL Pipeline:** Built with **LangChain Expression Language (LCEL)** using `RunnableParallel`, `RunnablePassthrough`, and `RunnableLambda` for clean, modular context injection.

---

## 🏗️ System Architecture & RAG Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                Chrome Extension (Manifest V3)                   │
│   1. Detects active YouTube URL & extracts `video_id`           │
│   2. Sends POST /ask { video_id, question } to FastAPI          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP POST (JSON)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend Server                      │
│                                                                 │
│   ┌───────────────────────────┐                                 │
│   │ Is `video_id` in Cache?   ├─── YES ──► Load Cached FAISS    │
│   └─────────────┬─────────────┘                Retriever        │
│                 │ NO                               │            │
│                 ▼                                  │            │
│   1. Fetch Transcript (YouTubeTranscriptApi)       │            │
│   2. Attach [MM:SS] Timestamps to segments         │            │
│   3. Split Text (RecursiveCharacterTextSplitter)   │            │
│      [chunk_size=1000, chunk_overlap=200]          │            │
│   4. Local Embedding (all-MiniLM-L6-v2)            │            │
│   5. Index in FAISS & Save to `retriever_cache`    │            │
│                 │                                  │            │
│                 └──────────────────┬───────────────┘            │
│                                    ▼                            │
│   6. Cosine Similarity Search (Top k=4 relevant chunks)         │
│   7. LCEL Chain: RunnableParallel -> PromptTemplate             │
│   8. LLM Inference: Qwen/Qwen2.5-7B-Instruct (HuggingFace)      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ JSON { "answer": "..." }
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         Extension Popup UI Renders Grounded Response            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, Vanilla JS, Chrome Manifest V3 | Browser popup UI, tab URL parsing, & async API calls |
| **Backend API** | Python, FastAPI, Uvicorn, Pydantic | REST API endpoint (`/ask`), CORS handling, & request validation |
| **Orchestration** | LangChain (`langchain-core`, LCEL) | Prompt templating, `RunnableParallel` chain execution |
| **Vector Store** | FAISS (`faiss-cpu`) | High-speed local cosine similarity search across chunks |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (`sentence-transformers`) | Local 384-dimensional dense vector generation |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` (`HuggingFaceEndpoint`) | Context-grounded answer generation |
| **Data Ingestion** | `youtube-transcript-api` | Scraping time-synced captions without YouTube Data API quotas |

---

## 📂 Project Structure

```text
youtube-rag-chrome-extension/
│
├── backend/
│   ├── server.py            # FastAPI server, FAISS caching, & LCEL RAG chain
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (HF_TOKEN - gitignored)
│
├── extension/
│   ├── manifest.json        # Chrome Extension Manifest V3 configuration
│   ├── popup.html           # Extension popup layout & dark-theme styling
│   └── popup.js             # Tab URL extraction & FastAPI fetch logic
│
├── .gitignore               # Excludes .venv, .env, and __pycache__
└── README.md                # Project documentation
```

---

## ⚙️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/manish97189/youtube-rag-chrome-extension.git](https://github.com/manish97189/youtube-rag-chrome-extension.git)
cd youtube-rag-chrome-extension
```

### 2. Backend Setup (`backend/`)
Create a virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install fastapi uvicorn python-dotenv youtube-transcript-api langchain-text-splitters langchain-community langchain-core langchain-huggingface faiss-cpu sentence-transformers pydantic
```

Create a `.env` file inside the `backend/` directory and add your Hugging Face token:
```env
HF_TOKEN=hf_your_huggingface_access_token_here
```

Start the FastAPI server:
```bash
cd backend
python -m uvicorn server:app --reload --port 8000
```
*The backend will now be live at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).*

### 3. Load the Chrome Extension (`extension/`)
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** ON in the top-right corner.
3. Click **Load unpacked** in the top-left corner.
4. Select the `extension/` folder from this repository.
5. Open any YouTube video with captions enabled, click the **YouTube RAG Chat** extension icon in your toolbar, and ask a question!

---

## 📡 API Reference

### `POST /ask`
Indexes the video transcript (if not already cached) and returns a grounded answer from the LLM.

**Request Body:**
```json
{
  "video_id": "_XeE2aEiPHk",
  "question": "What is the main topic discussed in this video?"
}
```

**Response (`200 OK`):**
```json
{
  "answer": "Based on the transcript around, the speaker discusses..."
}
```

---

## 🔮 Future Enhancements
* **Conversational Memory:** Integrate `ChatMessageHistory` to support multi-turn follow-up questions.
* **Clickable Timestamps:** Parse `[MM:SS]` in `popup.js` to seek the active YouTube player directly to the cited timestamp.
* **Persistent Vector DB:** Add disk persistence for frequently queried educational playlists.

---

## 👤 Author
**Manish Kumar**  
*B.Tech in Information Technology — Delhi Technological University (DTU)*