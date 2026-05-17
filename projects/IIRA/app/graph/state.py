from typing import List, TypedDict

class ResearchState(TypedDict):
	question: str
	plan: List[str]
	report_sections: List[str]
	final_report: str
	feedback: str
	loop_count: int