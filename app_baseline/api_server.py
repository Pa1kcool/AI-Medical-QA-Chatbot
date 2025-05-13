from fastapi import FastAPI
from pydantic import BaseModel
from inference import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/qa")
def qa_endpoint(query: Query):
    answer = answer_question(query.question)
    return {"question": query.question, "answer": answer}
