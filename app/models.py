from pydantic import BaseModel
from typing import List

# --------- 資料模型 (原 /execute 用) ---------
class ChatMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    phone_number: str
    messages: List[ChatMessage]