from fastapi import FastAPI
from app.models import UserQuery
from app.agent_service import execute_agent


app = FastAPI()

@app.post("/execute")
def execute(user_input: UserQuery):
    return execute_agent(user_input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)



# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# from agent import DoctorAppointmentAgent
# from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# import os
# from typing import List, Union # 引入 Union 以處理不同類型的消息

# app = FastAPI()

# class ChatMessage(BaseModel):
#     role: str
#     content: str

# # 定義 Pydantic 模型來接收請求主體
# class UserQuery(BaseModel):
#     phone_number: str
#     messages: List[ChatMessage] 


# agent = DoctorAppointmentAgent()

# @app.post("/execute")
# def execute_agent(user_input: UserQuery):
#     app_graph = agent.workflow() 
#     langchain_messages: List[BaseMessage] = []
#     for msg in user_input.messages:
#         if msg.role == 'user':
#             langchain_messages.append(HumanMessage(content=msg.content))
#         elif msg.role == 'assistant':
#             langchain_messages.append(AIMessage(content=msg.content))

#     query_data = {
#         "messages": langchain_messages, # 傳遞完整的 LangChain 訊息歷史
#         "phone_number": user_input.phone_number,
#         "next": "",
#         "query": "", # 這裡可以設置為 `langchain_messages[-1].content` 如果您想在進入 supervisor 之前就設置
#         "current_reasoning": "",
#     }

#     config = {"configurable": {"thread_id": str(user_input.phone_number), "recursion_limit": 20}}  

#     response = app_graph.invoke(query_data, config=config)
    
#     # 提取最後的 AI 訊息作為回應傳回前端
#     final_ai_message_content = "No response from agent."
#     if "messages" in response and response["messages"]:
#         for msg in reversed(response["messages"]): # 從後往前找
#             if isinstance(msg, AIMessage):
#                 final_ai_message_content = msg.content
#                 break
    
#     return {"messages": final_ai_message_content} # 只返回最新的 AI 回應內容