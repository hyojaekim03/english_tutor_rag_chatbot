# 🧠 ESL RAG Chatbot

A **RAG** project that helps ESL (English as a Second Language) students practice engaing in natural sounding conversations, and receive context-aware grammar and vocabulary guidance.

## 🏗️ Architecture
- **Frontend:** React.js, TypeScript (Currently Implementing)
- **Backend:** Python (FastAPI)  
- **LLM Provider:** OpenAI (gpt-4o model) (customizable)
- **Embedding Model:** OpenAI text-embedding-ada-002 (customizable)
- **Vector Store:** Pinecone vector DB


## 📝 History 
For a brief demo of a mock conversation with the chatbot, please look under /backend/history/conversation1.txt :)


## ⚙️ Installation (Backend)

1. Clone the repository:
   ```bash
   git clone https://github.com/hyojaekim03/english_tutor_rag_chatbot.git
   cd english_tutor_rag_chatbot/backend
   ```
2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
3. Set your environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export PINECONE_API_KEY="your-key-here"
   ```
4. Run server
   ```bash
   python app.py
   ```
