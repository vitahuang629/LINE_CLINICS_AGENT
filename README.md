# Line Clinics Agent

一個醫美診所專用的 LINE Chatbot，結合 **OpenAI / LangChain / LangGraph / LangSmith**，可以：
- 提供症狀諮詢與同理心回覆
- 推薦本診所療程
- 協助用戶預約並通知管理群組

---

## 功能特色
- AI 諮詢助理，能理解使用者症狀與需求
- 動態追問收集資訊，提供精準療程推薦
- 支援預約流程，自動整理預約資訊並通知 LINE 群組
- 使用 LangChain 與 OpenAI Embeddings 進行資料檢索

---

## 環境需求
- Python 3.11
- 建議使用虛擬環境 `venv`

---

## 資料搜尋比對
本專案使用 Retriever 結合兩種檢索方式：
1. 向量檢索（Vector Retriever）
使用 Chroma + OpenAI Embeddings 將診所療程 CSV 資料轉成向量，搜尋最相關的療程內容，
存放於 ./chroma_token_split 資料夾，第一次建立後會快取。
2. 關鍵字檢索（BM25 Keyword Retriever）
使用 BM25 演算法 對療程的 name、suitable_for、keywords 欄位加權搜尋，suitable_for 欄位加權三倍，讓匹配度提高，
使用 pickle 快取於 bm25.pkl，第二次執行直接載入快取。

-然後將 Vector Retriever 與 BM25 Keyword Retriever 組合權重設定為 [0.3, 0.7]，更偏向關鍵字匹配

---
