from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

import os
import pickle
import pandas as pd
import jieba
from utils.shared_resources import embedding_model, chinese_tokenizer
from typing import List


def reciprocal_rank_fusion(*ranked_lists, k=60) -> List[str]:
    scores = {}
    for rl in ranked_lists:
        for rank, doc_id in enumerate(rl, start=1):  # rank 從 1 開始 / Rank is 1-based
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # 排序並融合 / Sort fused scores
    return [d for d, _ in fused]

# 中文斷詞
# def chinese_tokenizer(text):
#     return [token for token in jieba.cut(text) if token.strip()]

def get_qa_retriever():
    doc = pd.read_csv("C:/Users/bexo6/OneDrive/桌面/line_clinics_agent/data/clinics_qa.csv", encoding="utf-8-sig")

    qa_documents_vector = []
    qa_documents_keyword = []

    for idx, row in doc.iterrows():
        # vector 用 (語意)
        vector_content = f"問題：{row['question']}\n答案：{row['answer']}\n"
        qa_documents_vector.append(
            Document(
                page_content=vector_content,
                metadata={"category": row["category"], "document_id": idx}
            )
        )

        # keyword 用 (重複 keywords 加強 BM25)
        weighted_keywords = (row["keywords"] + " ") * 3
        keyword_content = f"問題：{row['question']}\n答案：{row['answer']}\n關鍵字：{weighted_keywords}\n"
        qa_documents_keyword.append(
            Document(
                page_content=keyword_content,
                metadata={"category": row["category"], "document_id": idx}
            )
        )


    persist_dir = "./chroma_qa"
    if not os.path.exists(persist_dir):
        vectorstore = Chroma.from_documents(
            documents=qa_documents_vector,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
    else:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # --- BM25 retriever ---
    bm25_path = "./bm25_qa.pkl"
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        bm25_retriever = BM25Retriever.from_documents(
            qa_documents_keyword, preprocess_func=chinese_tokenizer, k=2
        )
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    # --- Ensemble (語意 + 關鍵字) ---
    qa_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 關鍵字比重低一點
    )

    return qa_retriever
    # def fused_retrieve(query: str, top_k: int = 5):
    #     # 取出各自結果
    #     bm25_results = bm25_retriever.get_relevant_documents(query)
    #     vector_results = vector_retriever.get_relevant_documents(query)

    #     # 取出 document_id 以利 RRF
    #     bm25_ids = [str(d.metadata["document_id"]) for d in bm25_results]
    #     vector_ids = [str(d.metadata["document_id"]) for d in vector_results]

    #     # 用 RRF 融合排序
    #     fused_ids = reciprocal_rank_fusion(bm25_ids, vector_ids)

    #     # 根據融合後的排序回傳 Document
    #     id_to_doc = {str(d.metadata["document_id"]): d for d in qa_documents_vector}
    #     fused_docs = [id_to_doc[i] for i in fused_ids[:top_k] if i in id_to_doc]

    #     return fused_docs
    
    # return fused_retrieve

# if __name__ == "__main__":
#     retriever = get_qa_retriever()
#     docs = retriever.get_relevant_documents("地址")
#     for i, doc in enumerate(docs):
#         print(f"Doc {i+1}: {doc.page_content}")

# if __name__ == "__main__":
#     retriever = get_qa_retriever()
#     docs = retriever("地址?")  # ← 直接呼叫 retriever()，不是 .get_relevant_documents()
#     for i, doc in enumerate(docs):
#         print(f"Doc {i+1}: {doc.page_content}")