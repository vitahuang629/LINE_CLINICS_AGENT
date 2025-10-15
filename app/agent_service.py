from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from .models import UserQuery
from agent import DoctorAppointmentAgent
from typing import List

agent = DoctorAppointmentAgent()

def execute_agent(user_input: UserQuery):
    app_graph = agent.workflow()
    langchain_messages: List[BaseMessage] = []
    for msg in user_input.messages:
        if msg.role == 'user':
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == 'assistant':
            langchain_messages.append(AIMessage(content=msg.content))

    query_data = {
        "messages": langchain_messages,  # 傳遞完整的 LangChain 訊息歷史
        "phone_number": user_input.phone_number,
        "next": "",
        "query": "", # 這裡可以設置為 `langchain_messages[-1].content` 如果您想在進入 supervisor 之前就設置
        "current_reasoning": "",
    }

    config = {"configurable": {"thread_id": str(user_input.phone_number), "recursion_limit": 20}}
    response = app_graph.invoke(query_data, config=config)

    final_ai_message_content = "No response from agent."
    if "messages" in response and response["messages"]:
        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage):
                final_ai_message_content = msg.content
                break

    return {"messages": final_ai_message_content}  # 只返回最新的 AI 回應內容