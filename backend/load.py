import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# pinecone v3 setup
pc = Pinecone(api_key=config.PINECONE_KEY)
index_name = "my-rag-index"
namespace = "default"
embedding_dim = 1536  # for text-embedding-ada-002

# create index not already created (safety net)
if index_name not in [i.name for i in pc.list_indexes()]:
    print(f"Creating Pinecone index '{index_name}' with dimension {embedding_dim}")
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# init embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def index_folder_with_metadata(folder_path: str, topic_name: str):
    all_chunks = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".pdf"):
            print(f"loading pdf: {file_path}")
            loader = PyPDFLoader(file_path)

        elif filename.endswith(".txt"):
            print(f"loading txt: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")

        else:
            continue  # skip unsupported file types

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata = {
                "source_file": filename,
                "topic": topic_name
            }
        all_chunks.extend(chunks)

    print(f"total Chunks: {len(all_chunks)}")

    texts = [doc.page_content for doc in all_chunks]
    metadatas = [doc.metadata for doc in all_chunks]
    embeddings = embedding_model.embed_documents(texts)

    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": embeddings[i],
            "metadata": {
                **metadatas[i],
                "text": texts[i]  
            }
        }
        for i in range(len(embeddings))
    ]

    # upload
    index = pc.Index(index_name)
    index.delete(delete_all=True, namespace=namespace)

    index.upsert(vectors=vectors, namespace=namespace)

    print(f"uploaded {len(vectors)} vectors to '{index_name}' (namespace: '{namespace}')")


# index command
if __name__ == "__main__":
    index_folder_with_metadata("./data/E2-Speech", topic_name="E2-Speech")
