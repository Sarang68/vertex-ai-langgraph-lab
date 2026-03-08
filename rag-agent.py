# agents/rag_agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator, json

# ── State ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    retrieved_chunks: list[dict]
    final_answer: str | None
    iteration: int
    access_groups: list[str]

# ── Tools ─────────────────────────────────────────────
@tool
def search_documents(query: str, top_k: int = 8) -> str:
    """Search the document corpus. Returns relevant passages."""
    # In real lab: call HybridRetriever.retrieve()
    return json.dumps([{"chunk_id": "c1", "text": "Sample retrieved content...",
                         "score": 0.92}])

@tool
def fetch_structured_data(entity_id: str, data_type: str) -> str:
    """Fetch structured data from internal APIs (inventory, pricing, etc.)."""
    return json.dumps({"entity_id": entity_id, "data": {"status": "active"}})

@tool
def validate_answer(answer: str, sources: str) -> str:
    """Reflection tool: check if answer is fully grounded in sources."""
    # Lightweight LLM call to score faithfulness
    return json.dumps({"grounded": True, "score": 0.95})

tools = [search_documents, fetch_structured_data, validate_answer]
llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0).bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iteration": state["iteration"] + 1}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["iteration"] >= 5:  # guardrail: max iterations
        return "end"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# ── Graph ─────────────────────────────────────────────
def build_rag_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue,
        {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=MemorySaver())

# ── Run ───────────────────────────────────────────────
agent = build_rag_agent()
result = agent.invoke({
    "messages": [HumanMessage("What is our refund policy for enterprise clients?")],
    "query": "refund policy enterprise",
    "retrieved_chunks": [], "final_answer": None,
    "iteration": 0, "access_groups": ["enterprise"]
})
