import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from typing import Type
from dotenv import load_dotenv

# from langchain_ollama import ChatOllama
load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY


# ─── LLM 呼叫的逾時與重試（全專案共用，勿在各處各寫一個數字）───────────────
# ⚠️ 不顯式指定 timeout 的話，langchain_openai 會把 None 一路傳給 httpx，
#    實際結果是 Timeout(timeout=None) —— 不是「等很久才失敗」，是**永遠不會返回**。
#
# 最常見的觸發情境不是 OpenAI 掛掉（那會很快回錯誤碼），而是連線被中間的
# NAT / 防火牆靜默切斷：沒有 RST，兩端都以為連線還活著，呼叫端就一直等。
#
# 代價很具體：/chat 是同步的、跑在 40 條 threadpool thread 上（見 main_webhook.py），
# 一個卡死的呼叫就永久佔住一條 thread。累積滿 40 條，那個 uvicorn worker 就再也收不了
# 新請求 —— 但進程還活著、port 還開著、/health 還是回 200（它跑在 event loop 上，
# 不經過 threadpool），從外面完全看不出來。
#
# 設了 timeout 之後，卡住會變成 APITimeoutError 拋出來，而
# app/backend_agent_service.py 的 🛡️ 兜底 try/except 本來就接得住
# （首次崩潰回罐頭請客人重述、再崩才轉真人）——
# 等於用一個參數把既有的降級機制真正啟用（在此之前那段 except 幾乎不會被觸發）。
#
# 數字取捨：單次呼叫最多等 60 秒，逾時／連線錯誤最多再重試 1 次
#   → 單一 LLM 呼叫最壞約 120 秒。
# graph 一輪會打好幾次 LLM，所以**不要再往上調**；要調的話記得乘上一輪的呼叫次數，
# 那才是客人實際等待時間的上限。
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 1


class LLMModel:
    """
    temperature 預設 0：本專案多數用途是「判斷/分類」（路由、planner、守門、審查、事實核對），
    這類任務不需要創意，設 0 可大幅降低同一輸入產生不同判斷的抽風情形。
    僅「客服回覆生成（Composer）」會刻意調高一點，保留一些自然語感。
    ⚠️ 不傳 temperature 時 ChatOpenAI 不會送這個參數 → 套用 OpenAI 預設 1.0（最大隨機性），
       因此這裡一律顯式指定，不要改回省略。
    """
    def __init__(
        self,
        model_name="gpt-4o",
        temperature: float = 0,
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.temperature = temperature
        self.openai_model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            timeout=timeout,          # ⚠️ 見上方 LLM_TIMEOUT 說明，不要移除
            max_retries=max_retries,
        )

    def get_model(self):
        return self.openai_model

# class LLMModel: gpt-4o-mini
#     """
#     使用 LangChain 的 ChatOllama 與自部署的 Ollama 模型進行互動。
#     這個類別封裝了模型的初始化，並使其與 LangChain 的功能 (如 with_structured_output) 相容。
#     """
#     def __init__(self, model_name="ycchen/breeze-7b-instruct-v1_0", ollama_base_url=OLLAMA_URL, use_json_format = False ):
#         """
#         初始化 LLMModel。

#         Args:
#             model_name (str): 要在 Ollama 中使用的模型名稱。
#             ollama_base_url (str): Ollama 伺服器的基礎 URL。
#         """
#         if not model_name:
#             raise ValueError("模型名稱 (model_name) 不得為空。")
#         if not ollama_base_url:
#             raise ValueError("Ollama 基礎 URL (ollama_base_url) 不得為空。")
            
#         self.model_name = model_name
#         self.ollama_base_url = ollama_base_url
        
#         self.model = ChatOllama(
#             model=self.model_name,
#             base_url=self.ollama_base_url,
#             temperature=0.7,
#             format="json" if use_json_format else None

#         )

#     def get_model(self):
#         """
#         返回已初始化的 LangChain ChatOllama 模型實例。
#         這樣外部程式碼就可以直接使用這個與 LangChain 相容的物件。

#         Returns:
#             ChatOllama: LangChain 的 ChatOllama 模型實例。
#         """
#         return self.model

# if __name__ == "__main__":
#     llm_instance = LLMModel()  
#     llm_model = llm_instance.get_model()
#     response=llm_model.invoke("hi")

#     print(response)

# if __name__ == "__main__":
#     # 正確呼叫：使用 JSON 輸出格式
#     llm_instance = LLMModel(use_json_format=True)
#     llm_model = llm_instance.get_model()

#     # 請提供結構化輸出的 prompt，否則會得到空字典 {}
#     prompt = """
#     請輸出以下格式的 JSON：
#     {
#       "greeting": "打招呼的句子",
#       "help": "你能提供的幫助"
#     }
#     """

#     response = llm_model.invoke(prompt)
#     print(response.content)