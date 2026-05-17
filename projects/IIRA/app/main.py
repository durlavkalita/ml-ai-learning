from fastapi import FastAPI
from graph.workflow import create_app

app = FastAPI()
agent = create_app()

@app.get("/")
def hello_world():
    return {"hello world"}

@app.post("/research")
async def run_research(user_query: str):
    inputs = {"question": user_query}
    result = await agent.ainvoke(inputs)
    return {"report": result["final_report"]}