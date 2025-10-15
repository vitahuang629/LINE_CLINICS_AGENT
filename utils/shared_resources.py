from langchain.embeddings import OpenAIEmbeddings
import jieba
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",   # 或 "text-embedding-3-large"
    api_key=OPENAI_API_KEY
)
#中文分詞
def chinese_tokenizer(text):
    return [token for token in jieba.cut(text) if token.strip()]