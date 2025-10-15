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

class LLMModel:
    def __init__(self, model_name="gpt-4.1-mini-2025-04-14"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.openai_model=ChatOpenAI(model=self.model_name)
        
    def get_model(self):
        return self.openai_model

# class LLMModel: gpt-4o-mini
#     """
#     使用 LangChain 的 ChatOllama 與自部署的 Ollama 模型進行互動。
#     這個類別封裝了模型的初始化，並使其與 LangChain 的功能 (如 with_structured_output) 相容。
#     """
#     def __init__(self, model_name="ycchen/breeze-7b-instruct-v1_0", ollama_base_url="http://3.115.142.115:11434", use_json_format = False ):
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