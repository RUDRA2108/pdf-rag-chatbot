# 📚 PDF RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://www.gradio.app/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Chat with your PDFs using **Retrieval-Augmented Generation (RAG)** – **completely local**.  
> Everything runs on your own machine using [Ollama](https://ollama.ai/). No cloud, no external API calls.  
> Extracts **text, tables, and images** (with captions) from long documents, indexes them, and lets you query them locally with an LLM.

---

## ✨ Features
- 🖥️ **Runs fully offline** – all models run locally via Ollama.  
- 📑 **Multi-PDF ingestion** – process thousands of pages at once.  
- 🔍 **RAG-powered search** – retrieve relevant context before answering.  
- 📊 **Table & image support** – tables are converted to Markdown, images captioned with a vision model.  
- 💬 **Interactive chatbot UI** built with Gradio.  
- 📥 **Export chat history to PDF**.  
- ⚡ **Local-first** – your data never leaves your computer.  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/RUDRA2108/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create & activate virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
Copy the example file and edit if needed:
```bash
cp .env.example .env
```

Default `.env` values:
```env
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=llama3:8b
VISION_MODEL=llava
CHROMA_DIR=vectorstore
```

⚠️ Make sure [Ollama](https://ollama.ai/) is installed and running locally on your machine.  
No internet connection is required once models are downloaded.

---

## 📂 Usage

### Step 1: Ingest PDFs
Place your PDF files inside the `Documents/` folder, then run:
```bash
python ingest.py --recursive
```
This will parse PDFs → extract text, tables, images → save results into `output/`.

### Step 2: Launch the chatbot
```bash
python app.py
```
Open your browser at **http://localhost:7860**.  
You can now chat with your documents **fully locally**.

### Step 3: Export chat
Click the **📥 Download Chat as PDF** button to save your conversation.

---

## 🛠 Project Structure
```
pdf-rag-chatbot/
├── app.py           # Chatbot app (Gradio + RAG)
├── ingest.py        # PDF ingestion pipeline
├── Documents/       # Place your PDFs here
├── output/          # Processed PDFs (Markdown + JSON + images/tables)
├── vectorstore/     # ChromaDB storage
├── .env.example     # Example environment file
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
