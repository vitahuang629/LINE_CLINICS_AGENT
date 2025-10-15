
from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from prompt_library.prompt import system_prompt
from utils.llms import LLMModel
from toolkit.toolkits import *
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import json
from utils.profile_db import get_user_profile_by_phone


def get_latest_human_message(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return ""


# def is_end(state):
#     user_msg = get_latest_human_message(state["messages"])
#     if any(kw in user_msg for kw in ['謝謝', '了解', '沒事']):
#         return END
#     return "supervisor"

# def user_wants_booking(user_query: str) -> bool:
#     """
#     簡單判斷使用者是否想要預約
#     """
#     booking_keywords = ["預約", "幫我安排", "我要報名", "想體驗", "想試試", "可以預約"]
#     return any(kw in user_query for kw in booking_keywords)

def format_user_profile_text(user_profile_dict: dict) -> str:
    lines = ["使用者個人資料如下："]
    for category, entries in user_profile_dict.items():
        for entry in entries:
            # 支援一個類別多筆內容
            lines.append(f"- {category}: {entry}")
    return "\n".join(lines)

class Router(TypedDict):   #openai
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str

class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    phone_number: str
    next: str
    query: str
    current_reasoning: str
    booking_completed: bool  #0716 
    should_terminate: bool  #
    user_profile: dict
    is_new_customer: bool


class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel() #openai
        self.llm_model=llm_model.get_model()  #openai
        # llm_model = LLMModel(use_json_format=True)
        # self.llm_model=llm_model.get_model()

    def start_node(self, state: AgentState):
        print("start_node called")

        phone_number = state["phone_number"]
        user_profile = get_user_profile_by_phone(phone_number)
        state["user_profile"] = user_profile
        state["is_new_customer"] = user_profile is None or not user_profile

        print('sssssssssssssss', state["is_new_customer"])

        return {
            "user_profile": user_profile,
            "phone_number": phone_number,
            "messages": [],
            "is_new_customer": state["is_new_customer"],
            "next": "supervisor"
        }
    
    # supervisor_node 修正
    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
        if not state["messages"]:
            print("Error: messages list is empty in supervisor_node")
            return Command(goto=END, update={'messages': [AIMessage(content="對不起，我沒有收到您的查詢。")]}) 

        current_user_query = get_latest_human_message(state["messages"])

        system_prompt = """
        您是一位「醫美診所經理」，負責管理專業助理（workers）協作。  
        工作分配規則：

        - WORKER: information_node  
        功能：提供療程相關資訊、解答症狀或需求問題。

        - WORKER: booking_node  
        功能：負責與預約或診所服務資訊有關的問題，如「預約、改期、取消、費用、電話、地址、初診」等。

        - WORKER: FINISH  
        功能：對話結束。

        判斷原則：

        1. 使用者如果只是想詢問療程內容或健康問題，請回傳 {"next": "information_node", "reasoning": "...理由..."}。  
        2. 使用者訊息包含「費用」「價錢」「價格」「初診」「地址」「電話」「預約」「改期」「取消」「時間」等字眼，請回傳 {"next": "booking_node", "reasoning": "...理由..."}。  
        3. 使用者表示問題已解決或想結束，請回傳 {"next": "FINISH", "reasoning": "...理由..."}。

        範例：

        使用者：我想了解減重療程有哪些？  
        回覆：{"next": "information_node", "reasoning": "使用者詢問療程資訊"}

        使用者：我要預約下週一的療程  
        回覆：{"next": "booking_node", "reasoning": "使用者要求預約"}

        使用者：謝謝，沒其他問題了  
        回覆：{"next": "FINISH", "reasoning": "使用者表達結束對話"}

        請根據上述規則，判斷下一步應該指派給哪個專業助理，並說明理由。

        """
        
        print('currrrrrrrrrrrrrr', current_user_query)
        # openai           
        messages_for_llm = [
            {"role": "system", "content": system_prompt},
            HumanMessage(content=f"使用者個手機號碼是: {state['phone_number']}"),
            HumanMessage(content=f"使用者是新客: {state['is_new_customer']}"),
        ] + state["messages"] # 包含所有歷史訊息

        response = self.llm_model.with_structured_output(Router).invoke(messages_for_llm) # 使用修正後的 messages_for_llm

        print("supervisor_node response:", response)

        query = ''
        if len(state['messages']) == 1:
            query = state['messages'][0].content
        goto = response["next"]
        
        print("********************************this is my goto*************************")
        print(goto)
        
        print("********************************")
        print(response["reasoning"])
            
        if goto == "FINISH":
            return Command(
                goto=END,
                update={
                    'next': END,
                    'current_reasoning': response["reasoning"],
                    # 覆蓋舊訊息，確保結束時不會重播上一則
                    'messages': [AIMessage(content="感謝您的諮詢，如有任何問題，請隨時與我們聯繫。")]
                }
            )

        # 其他分支
        if query:
            return Command(goto=goto, update={
                'next': goto,
                'query': query,
                'current_reasoning': response["reasoning"]
            })

        return Command(goto=goto, update={
            'next': goto,
            'current_reasoning': response["reasoning"]
        })





    def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called information node************")


        raw_system_prompt = """
            你是一位專業且有同理心的醫美諮詢助理，使用者會輸入症狀或需求，例如「我失眠很嚴重」、「我最近痘痘變多」。

            You run in a loop of Thought, Action, PAUSE, Observation.
            At the end of the loop you output an Answer
            Use Thought to describe your thoughts about the question you have been asked.
            Use Action to run one of the actions available to you - then return PAUSE.
            Observation will be the result of running those actions.

            你可以使用的行動工具包括：
            - get_empathy_questions_by_symptom：取得針對使用者症狀的同理話語和追問句。
            - search_clinics_by_keyword：查詢並推薦適合的醫美療程。

            使用規則：
            - 每一回合最多呼叫一個工具。
            - 建議先用 get_empathy_questions_by_symptom，同理關心並追問，直到收集足夠資訊。
            - 若使用者明確表達想知道療程推薦，可直接呼叫 search_clinics_by_keyword。
            - 不要印出Thought, Action, PAUSE過程
            - 如果使用者重複提到相同的症狀（如皺紋、法令紋、木偶紋），不要重複使用相同的同理心語句
            - 當檢測到重複症狀時，改用簡短確認並提出新的追問問題

            請依照上述流程循環執行，直到你能完整回覆使用者需求。

            現在開始回答使用者的問題：

"""
        formatted_profile = format_user_profile_text(state["user_profile"])

        system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", raw_system_prompt),  # 給模型的角色說明
        ("ai", formatted_profile),      # 把使用者資料當作 AI 已知的事實
        ("ai", f"使用者是新客: {state['is_new_customer']}"),  # 同樣視為背景
        ("placeholder", "{messages}")   # 用戶輸入與歷史訊息
    ]
)
        # print('Prompt created:', system_prompt_template)

        information_agent = create_react_agent(model=self.llm_model,tools=[get_empathy_questions_by_symptom, search_clinics_by_keyword] ,prompt=system_prompt_template)

        # print("information_agent =", information_agent)


        result = information_agent.invoke({"messages": state["messages"]})
        # print('original_answser', result)
        print('resulttttttttttttt', result["messages"][-1].content)
        
        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=result["messages"][-1].content, name="information_node")
                    # HumanMessage(content=result["messages"][-1].content, name="information_node")
                ]
            },
        )

        
    def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("*****************called booking node************")
        
 
        system_prompt = """
            你是一位專業且有條理的醫美預約助理，負責幫助使用者：
            1. 預約療程
            2. 查詢診所資訊（例如地址、電話、初診費用）

            You run in a loop of Thought, Action, PAUSE, Observation.
            At the end of the loop you output an Answer
            Use Thought to describe your thoughts about the question you have been asked.
            Use Action to run one of the actions available to you - then return PAUSE.
            Observation will be the result of running those actions.

            你可以使用的行動工具包括：

            你可以使用的工具：
            - **set_appointment**：當使用者想要預約、安排時間或提供預約資訊時使用。
            - **search_clinics_info**：當使用者詢問診所「地址」、「位置」、「電話」、「初診費用」、「怎麼去」、「在哪裡」等時使用。  
            此工具只回傳資料庫的固定答案，不要自行生成內容。
            
            ---

            ### 使用規則
            - 如果使用者想要「預約療程」→ 使用 `set_appointment`
            - 如果使用者想問「診所資訊（地址、電話、費用）」→ 使用 `search_clinics_info`
            - 不要同時使用兩個工具。
            - 回答時不要自己想像或延伸回答，只能根據工具回傳內容回答。

            ---

            ### 範例
            使用者：診所在哪裡？
            AI：呼叫 search_clinics_info 工具，問題 = 診所地址
            使用者：電話？
            AI：呼叫 search_clinics_info 工具，問題 = 診所電話
            使用者：收費？
            AI：呼叫 search_clinics_info 工具，問題 = 初診費用
            """
        booking_agent = create_react_agent(model=self.llm_model, tools=[set_appointment, search_clinics_info], prompt=system_prompt)

        # print("🔍 booking_agent:", booking_agent)

        result = booking_agent.invoke(state) # 這裡 result["messages"] 包含了代理的輸出和可能的工具調用結果
        answer = result["messages"][-1].content

        # print("🔍 Agent invoke result:", result)

        print('resulttttttttttttt', answer)
        
        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=answer, name="booking_node")
                    # HumanMessage(content=result["messages"][-1].content, name="information_node")
                ]
            },
        )



    def workflow(self):
        memory = MemorySaver()
        self.graph = StateGraph(AgentState)
        self.graph.add_node("start_profilo", self.start_node)
        self.graph.add_node("supervisor", self.supervisor_node)
        self.graph.add_node("information_node", self.information_node)
        self.graph.add_node("booking_node", self.booking_node)
        # 修改起始節點指向 start
        self.graph.set_entry_point("start_profilo")
        self.graph.set_finish_point("supervisor")


        # 設定節點之間的連線
        self.graph.add_edge("start_profilo", "supervisor")
        self.graph.add_edge('supervisor', END)

        self.app = self.graph.compile(checkpointer = memory)
        return self.app
    

    # def workflow(self):
    #     memory = MemorySaver()
    #     self.graph = StateGraph(AgentState)

    #     self.graph.add_node("supervisor", self.supervisor_node)
    #     self.graph.add_node("information_node", self.information_node)
    #     self.graph.add_node("booking_node", self.booking_node)

    #     self.graph.add_edge(START, "supervisor")
    #     self.graph.add_edge("supervisor", "information_node")
    #     self.graph.add_edge("supervisor", "booking_node")
    #     self.graph.add_edge("information_node", "supervisor")
    #     self.graph.add_edge("booking_node", "supervisor")
    #     self.graph.add_edge("supervisor", END)

    #     self.app = self.graph.compile(checkpointer=memory)
    #     return self.app


            #     你是一位專業且有同理心的醫美諮詢助理，使用者會輸入症狀或需求，例如「我失眠很嚴重」、「我最近痘痘變多」。

            # 你工作的流程是個循環，包含以下階段：
            

            # - Thought：根據目前資訊，思考如何幫助使用者。
            # - Action (Pause)：選擇並執行一個可用工具，等待工具結果回傳。
            # - Observation：根據工具回覆，整合資訊準備下一輪思考。
            # - Answer：請直接給出回答，並結束對話。

            # 你可以使用的行動工具包括：
            # - get_empathy_questions_by_symptom：取得針對使用者症狀的同理話語和追問句。
            # - search_clinics_by_keyword：查詢並推薦適合的醫美療程。

            # 使用規則：
            # - 每一回合最多呼叫一個工具。
            # - 建議先用 get_empathy_questions_by_symptom，同理關心並追問，直到收集足夠資訊。
            # - 若使用者明確表達想知道療程推薦，可直接呼叫 search_clinics_by_keyword。

            # 請依照上述流程循環執行，直到你能完整回覆使用者需求。

            # 現在開始回答使用者的問題：


    # def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
    #     print("*****************called booking node************")
        
 
    #     system_prompt = """
    #         你是一位專業的療程預約員，使用者會輸入想要預約的日期和時間，例如「我想預約8/5痘痘」、「我想預約明天」，或是查詢可預約的時間，例如「7/22可以預約的時段」、「明天可以的時段」。

    #         You run in a loop of Thought, Action, PAUSE, Observation.
    #         At the end of the loop you output an Answer
    #         Use Thought to describe your thoughts about the question you have been asked.
    #         Use Action to run one of the actions available to you - then return PAUSE.
    #         Observation will be the result of running those actions.

    #         請注意：

    #         - 所有日期必須輸出成 MM-DD-YYYY 格式，月份放前面，若日期或月份小於10，請補零，例如 8月7日 ➜ 08-07-2025。
    #         - 請依使用者輸入，靈活調用工具，並且每次最多呼叫一個工具。

    #         你可以使用的行動工具包括：

    #         - check_availability: 確認可以預約的時間
    #         - set_appointment: 預約療程
    #         - cancel_appointment: 取消預約
    #         - reschedule_appointment: 療程改期

    #         請依上述流程循環執行，直到能完整回覆使用者的預約需求。

    #         現在開始回覆使用者的問題：
    #         """
    #     booking_agent = create_react_agent(model=self.llm_model, tools=[check_availability, set_appointment, cancel_appointment, reschedule_appointment], prompt=system_prompt)

    #     # print("🔍 booking_agent:", booking_agent)

    #     result = booking_agent.invoke(state) # 這裡 result["messages"] 包含了代理的輸出和可能的工具調用結果

    #     # print("🔍 Agent invoke result:", result)


    #     final_booking_message_content = "我已處理您的預約請求。請問還有其他需要嗎？" # 預設結束語
    #     booking_completed = False  # 新增完成標記
    #     should_terminate = False #7/16新增終止標記
    #     if result and "messages" in result:
    #         for msg in reversed(result["messages"]):
    #             # 終止條件1: 代理明確返回成功消息
    #             if isinstance(msg, AIMessage):
    #                 if any(keyword in msg.content for keyword in ["可預約的時間", "預約成功", "已完成", "取消成功", "已修改"]):
    #                     booking_completed = True
    #                     should_terminate = True
    #                 final_booking_message_content = msg.content
    #                 break 
    #             # 終止條件2: 工具返回成功結果
    #             elif isinstance(msg, HumanMessage) and msg.name == "tool_output":
    #                 if any(keyword in msg.content for keyword in ["Successfully", "成功", "完成", "已更新"]):
    #                     booking_completed = True
    #                     should_terminate = True
    #                 final_booking_message_content = f"預約已處理：{msg.content}. 還有其他需要嗎？"
    #                 break

    #             # 終止條件3: 檢測到錯誤或無法處理
    #             elif isinstance(msg, AIMessage) and any(keyword in msg.content for keyword in ["無法處理", "錯誤", "失敗"]):
    #                 should_terminate = True
    #                 break

    #     return Command(
    #         update={
    #             "messages": state["messages"] + [
    #                 AIMessage(content=final_booking_message_content, name="booking_node")
    #             ],
    #             "booking_completed": booking_completed,
    #             "should_terminate": should_terminate  # 新增狀態
    #         },
    #         goto="supervisor",
    #     )