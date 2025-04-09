import pinecone
import config

pinecone.init(api_key=config.PINECONE_KEY, environment=config.PINECONE_ENV)

index = pinecone.Index("my-rag-index")
index.delete(delete_all=True)

print("All vectors deleted from 'my-rag-index'")
