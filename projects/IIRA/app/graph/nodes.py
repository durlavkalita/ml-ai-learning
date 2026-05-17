from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from graph.state import ResearchState

load_dotenv()

search_tool = TavilySearch(max_results = 5)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

class Plan(BaseModel):
    """Series of research steps to answer a user's complex question"""
    steps: List[str] = Field(description="Individual research tasks or question to investigate")
planner_llm = llm.with_structured_output(Plan)

class ReviewResponse(BaseModel):
    decision: str = Field(description="Either 'PASS' or 'FAIL'")
    feedback: str = Field(description="Specific advice for the writer if FAIL")    
reviewer_llm = llm.with_structured_output(ReviewResponse)


def planner_node(state: ResearchState):
	"""Analyze the question and create a research plan"""
	prompt = f"""You are a research expert. 
	Break down the following complex query into 3-4 specific research tasks.
	Query: {state['question']}"""

	# llm response will be of Plan object
	response = planner_llm.invoke(prompt)

	return {"plan": response.steps}


def search_node(state: ResearchState):
    """Execute search for each step in the plan and stores in results"""
    all_results = []
    print(f"--- Executing Search for {len(state['plan'])} Steps ---")
    for step in state['plan']:
        print(f"Searching for: {step}")
        results = search_tool.invoke({"query": step})
        results = results['results'] 
        combined_content = "\n".join([r["content"] for r in results])
        all_results.append({'step': step, 'content': combined_content})
    return {"report_sections": all_results}


def writer_node(state: ResearchState):
    """Synthesize search results into a final report"""
    print("--- Generating final report ---")
    context = ""
    for entry in state['report_sections']:
        context += f"\n\nSource Topic: {entry['step']}\nContent: {entry['content']}"
    prompt = f"""You are a professional research analyst. 
    Based on the following research data, write a comprehensive report answering the original question.
    
    Original Question: {state['question']}
    
    Research Data:
    {context}
    "Earlier feedback for improvement: {state.get('feedback', 'None')}"
    Instructions:
    - Use Markdown formatting (headers, bullet points).
    - Be factual and concise.
    - If information is missing, acknowledge it.
    """
    response = llm.invoke(prompt)
    return {"final_report":  response.content}


def reviewer_node(state: ResearchState):
    """Review final report and provide feedback"""
    current_count = state.get("loop_count", 0)
    print("--- Reviewing final report ---")
    question = state['question']
    report = state['final_report']
    prompt = f"""You are a strict Senior Research Quality Auditor. 
	Your job is to criticize the following research report for accuracy, completeness, and structure.

	Original Question: {question}
	Report to Review: {report}

	Critically evaluate:
	1. Does this fully answer the user's question?
	2. Is the formatting professional?
	3. Are there any logical gaps?

	Output your response in this EXACT JSON format:
	{{
		"decision": "PASS" or "FAIL",
		"feedback": "If FAIL, list exactly what needs to be improved. If PASS, leave empty."
	}}
	"""
    response = reviewer_llm.invoke(prompt)
    return {"feedback": f"{response.decision}: {response.feedback}", "loop_count": current_count + 1}