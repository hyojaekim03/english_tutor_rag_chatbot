from fastapi import FastAPI
from pydantic import BaseModel
from query import rag_chain 

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    unit: str

@app.post("/query")
async def query_rag(req: QueryRequest):
    answer = rag_chain.invoke({
            "question": req.question,
            "unit": req.unit
            })
    return {"answer": answer}
