import requests
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
from typing import List
from tqdm import tqdm
import os
import jieba
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from langchain.schema import Document
from pydantic import Field
from langchain_openai import OpenAIEmbeddings
from utils.shared_resources import embedding_model, chinese_tokenizer



# #中文分詞
# def chinese_tokenizer(text):
#     return [token for token in jieba.cut(text) if token.strip()]

# # 🔧 你的 AWS Ollama embedding 端點
# OLLAMA_URL = OLLAMA_URL
# EMBED_MODEL = "bge-m3"

# # ✅ 自訂嵌入模型類別
# class OllamaEmbeddings(Embeddings):
#     """使用Ollama API的嵌入模型"""
    
#     def __init__(self, ollama_url: str = None, model: str = None):
#         self.ollama_url = ollama_url or OLLAMA_URL
#         self.model = model or EMBED_MODEL
    
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """嵌入文檔列表"""
#         return [self.embed_query(text) for text in texts]
    
#     def embed_query(self, text: str) -> List[float]:
#         """嵌入單個查詢"""
#         try:
#             response = requests.post(
#                 f"{self.ollama_url}/api/embeddings",
#                 json={"model": self.model, "prompt": text},
#                 timeout=60
#             )
#             response.raise_for_status()
#             embedding = response.json().get("embedding", [])
            
#             if not embedding:
#                 print(f"警告: 從API獲取到空向量，使用零向量替代")
#                 return [0.0] * 1024  # 根據您的模型調整維度
                
#             return embedding
#         except Exception as e:
#             print(f"嵌入查詢時發生錯誤: {e}")
#             return [0.0] * 1024  # 根據您的模型調整維度




def get_ensemble_retriever():
    # 讀取數據
    doc = pd.read_csv("data/clinics_introductions3.csv", encoding="big5")
    # print(doc)

    documents_for_vector = []
    documents_for_keyword = []


    for idx, row in doc.iterrows():
        # 正常版本（給 vector retriever）
        normal_content = (
            f"療程名稱：{row['name']}\n"
            f"介紹內容：{row['introduction']}\n"
            f"適合對象：{row['suitable_for']}\n"
            f"常見關鍵字：{row['keywords']}（例如使用者可能會說出這些詞）\n"   
        )
        
        # 加重版本（給 keyword retriever）
        suitable_for_weighted = (row['suitable_for'] + " ") * 3
        keyword_content = (
            f"療程名稱：{row['name']}\n"
            f"介紹內容：{row['introduction']}\n"
            f"適合對象：{suitable_for_weighted}\n"
            f"常見關鍵字：{row['keywords']}（例如使用者可能會說出這些詞）\n"
        )
        
        documents_for_vector.append(
            Document(page_content=normal_content, metadata={"clinic": row["name"],
                                                            "category": row["category"],
                                                            "document_id": idx})
        )
        documents_for_keyword.append(
            Document(page_content=keyword_content, metadata={"clinic": row["name"], 
                                                            "category": row["category"],
                                                            "document_id": idx})
        )

    persist_dir = "./chroma_token_split"  # 你要存放資料庫的資料夾路徑

    #如果這段重複執行, 會造成retriever重複, 第一次使用就好
    if not os.path.exists(persist_dir):
        vectorstore = Chroma.from_documents(
            documents=documents_for_vector,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
    else:
        print("✅ 資料庫已存在，略過建立步驟")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

    # 之後想用 retriever，就從本地載入
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


    # 建立/載入 BM25 keyword retriever（使用快取）
    bm25_path = "./bm25.pkl"

    if os.path.exists(bm25_path):
        print("✅ 載入快取的 BM25 Retriever")
        with open(bm25_path, "rb") as f:
            keyword_retriever = pickle.load(f)
    else:
        print("🚧 建立 BM25 Retriever 並快取")
        keyword_retriever = BM25Retriever.from_documents(documents_for_keyword, preprocess_func=chinese_tokenizer,k = 3)
        with open(bm25_path, "wb") as f:
            pickle.dump(keyword_retriever, f)


    ensemble_retriever = EnsembleRetriever(retrievers = [vector_retriever, keyword_retriever], weights = [0.3, 0.7])
    return ensemble_retriever

# if __name__ == "__main__":
#     retriever = get_ensemble_retriever()  # <== 要呼叫函式拿 retriever 回來
#     docs = retriever.get_relevant_documents("打呼")
#     for i, doc in enumerate(docs):
#         print(f"Doc {i+1}: {doc.page_content}")