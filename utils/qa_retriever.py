from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

import os
import pickle
import pandas as pd
import jieba
from utils.shared_resources import embedding_model, chinese_tokenizer

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
        weights=[0.2, 0.8]  # 關鍵字比重高一點
    )

    return qa_retriever

# if __name__ == "__main__":
#     retriever = get_qa_retriever()
#     docs = retriever.get_relevant_documents("地址")
#     for i, doc in enumerate(docs):
#         print(f"Doc {i+1}: {doc.page_content}")
