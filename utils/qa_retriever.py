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

def get_qa_retriever(
    csv_path: str = "data/clinics_qa.csv",
    persist_dir: str = "./chroma_qa",
    bm25_path: str = "./bm25_qa.pkl",
    k: int = 2,
):
    """建立 QA ensemble retriever（向量 + BM25）。

    參數化是為了讓「療程內容 QA」與「診所交易型 QA」各用一份獨立索引：
    同一段建索引邏輯共用，但每個來源檔有自己的 persist_dir / bm25_path，
    避免兩個領域的資料灌進同一個索引互相污染檢索排名。
    ⚠️ 不同來源務必給不同的 persist_dir 與 bm25_path，否則快取會互蓋。
    """
    doc = pd.read_csv(csv_path, encoding="utf-8-sig")

    qa_documents_vector = []
    qa_documents_keyword = []

    for idx, row in doc.iterrows():
        # 空值防呆：留空的格子在 pandas 是 NaN，直接 str() 會變成 "nan" 污染檢索，
        # 這裡統一轉成空字串。
        category = "" if pd.isna(row["category"]) else str(row["category"]).strip()
        keywords = "" if pd.isna(row["keywords"]) else str(row["keywords"]).strip()

        # 分類欄（療程專屬問答時 = 療程名，如「冷凍減脂」）也拼進可搜尋文字，
        # 否則同一個問題（如「效果如何」）在不同療程間分不出來、會撈錯筆。
        # vector 用 (語意)
        vector_content = f"分類：{category}\n問題：{row['question']}\n答案：{row['answer']}\n"
        qa_documents_vector.append(
            Document(
                page_content=vector_content,
                metadata={"category": row["category"], "document_id": idx}
            )
        )

        # keyword 用 (重複 keywords 加強 BM25；分類加權 ×2 幫助分辨是哪個療程)
        weighted_keywords = (keywords + " ") * 3
        weighted_category = (category + " ") * 2
        keyword_content = f"分類：{weighted_category}\n問題：{row['question']}\n答案：{row['answer']}\n關鍵字：{weighted_keywords}\n"
        qa_documents_keyword.append(
            Document(
                page_content=keyword_content,
                metadata={"category": row["category"], "document_id": idx}
            )
        )


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
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # --- BM25 retriever ---
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        bm25_retriever = BM25Retriever.from_documents(
            qa_documents_keyword, preprocess_func=chinese_tokenizer, k=k
        )
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    # 快取的 pickle 會把舊的 k 一起帶回來 → 載入後強制覆寫，確保 k 生效（不必刪快取重建）
    bm25_retriever.k = k

    # --- Ensemble (語意 + 關鍵字) ---
    qa_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 關鍵字比重低一點
    )

    return qa_retriever

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