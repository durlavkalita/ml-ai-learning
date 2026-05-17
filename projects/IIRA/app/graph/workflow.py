from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
from graph.nodes import planner_node, search_node, writer_node, reviewer_node

def create_app():
    def should_continue(state: ResearchState):
        """Router: Decides whether to finish or rewrite."""
        if state.get("loop_count", 0) >= 3:
            print("--- MAX ITERATIONS REACHED: FORCING END ---")
            return END
        if "PASS" in state["feedback"]:
            return END
        else:
            print(f"--- FEEDBACK: {state['feedback']} ---")
            return "writer" # Loop back to writer

    # Building the Graph
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("search", search_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search")
    builder.add_edge("search", "writer")
    builder.add_edge("writer", "reviewer")

    # This is where the loop happens!
    builder.add_conditional_edges(
        "reviewer", 
        should_continue, 
        {
            END: END,           # If should_continue returns END, go to END
            "writer": "writer"  # If should_continue returns "writer", go to writer
        }
    )

    return builder.compile()