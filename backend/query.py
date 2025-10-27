import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pinecone import Pinecone
import config
import uuid

# env vars
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# init Pinecone v3 client
pc = Pinecone(api_key=config.PINECONE_KEY)
index = pc.Index("my-rag-index")
namespace = "default"

# same embedding model as doc loaders
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")


# prompt
prompt = PromptTemplate(
    input_variables=["context", "question", "unit"],
    template="""
I want you to act as a friendly English teacher teaching young international students. I will speak to you in English about a textbook topic and you will reply to me in English to practice my spoken English. I want you to keep your reply neat, limiting the reply to 100 words. I want you to gently correct my grammar mistakes, typos, and factual errors that contradict the textbook. You can ask me a questions if it doesn't interrupt the flow of conversation, but keep it natural. Don't be too abrupt with questions. Now let’s start practicing. Remember, I want you to strictly correct my grammar mistakes, typos, and factual errors.
Use the following textbook context to answer the question. If you don’t know, say you don’t know. However, don't be too demanding in your answers because they are young children that can make mistakes. Be brief in your corrections.
Keep the tone light and playful to make the conversation enjoyable.
Start the conversation by briefly introducing the topic from the {unit} Passage, then ask a fun or interesting question about it.

Textbook Context:
{context}

Question:
{question}
"""
)

# llm
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
parser = StrOutputParser()

# search
def retrieve_relevant_chunks(question, top_k=3):
    question_embedding = embedding_model.embed_query(question)
    results = index.query(vector=question_embedding, top_k=top_k, namespace=namespace, include_metadata=True)

    print('top_k results: ', results)

    chunks = [match["metadata"]["text"] for match in results["matches"] if "text" in match["metadata"]]
    return "\n\n".join(chunks) if chunks else "No relevant context found."

# RAG chain
rag_chain: Runnable = (
    {
        "context": lambda x: retrieve_relevant_chunks(x["question"]),
        "question": lambda x: x["question"],
        "unit": lambda x: x["unit"]
    }
    | prompt
    | llm
    | parser
)

# query
if __name__ == "__main__":
    unit = "E2B-Speech-Unit-1"
    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break
        answer = rag_chain.invoke({
            "question": question,
            "unit": unit
            })

        print("\n Answer:", answer)

        with open("./history/chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {question}\n")
            f.write(f"AI: {answer}\n\n")
