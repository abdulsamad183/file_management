from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from fm_tools import create_folder, list_folder, create_file, edit_file, rename, delete

from dotenv import load_dotenv
load_dotenv()


MODEL_NAME = "openai:gpt-4o-mini"

# --------------------
# LLM Node wrapper
# --------------------
def make_llm_node():
    tools = [create_folder, list_folder, create_file, edit_file, rename, delete]
    llm = init_chat_model(MODEL_NAME).bind_tools(tools)
    def llm_node(state: MessagesState):
        system_msg = [
            HumanMessage(role="system", content=(
                "You are an AI File Manager operating ONLY in the './workspace' sandbox.\n"
                "Whenever the user asks to create, edit, rename, delete, or list a file/folder, "
                "you MUST call the appropriate tool immediately instead of explaining steps.\n"
                "Never describe how to do it manually — just use the tools."
            ))
        ]
        msgs = system_msg + state["messages"]
        return {"messages": llm.invoke(msgs)}
    return llm_node, tools

# --------------------
# Build Graph
# --------------------
def build_agent():
    llm_node, tools = make_llm_node()
    tool_node = ToolNode(tools=tools)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("agent", llm_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)

    graph_builder.set_entry_point("agent")
    graph_builder.set_finish_point("agent")
    return graph_builder.compile()

# --------------------
# Run agent
# --------------------
def run_prompt(compiled_graph, user_prompt: str):
    human = HumanMessage(content=user_prompt)
    result_state = compiled_graph.invoke({"messages": [human]})
    messages = result_state.get("messages", [])

    out_lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.content:
            out_lines.append(f"AI: {m.content}")
        elif isinstance(m, ToolMessage):
            out_lines.append(f"TOOL RESULT ({m.name}): {m.content}")

    return "\n".join(out_lines) or "(no response)"

# --------------------
# Main Function
# --------------------
save_graph = False
def main():
    print("Starting LangGraph AI File Manager (sandbox ./workspace)")
    compiled = build_agent()

    if save_graph: # Save Mermaid diagram to file
        with open("graph.mmd", "w") as f:
            f.write(compiled.get_graph().draw_mermaid()) # It can be viewed in any mermaid editor

    print("Agent ready. Type natural commands (e.g., 'create a folder reports', 'create file reports/t1.txt with hello')")
    print("Type 'exit' to quit, 'ls' to list workspace root.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not user:
            continue
        if user.lower() in ("exit", "quit", "bye","see you"):
            print("bye.")
            return
        if user.lower() == "ls":
            print(list_folder("."))
            continue

        try:
            out = run_prompt(compiled, user)
        except Exception as e:
            out = f"Error running agent: {e}"
        print("\n" + out)

if __name__ == "__main__":
    main()
