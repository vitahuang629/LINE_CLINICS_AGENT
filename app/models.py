from pydantic import BaseModel
from typing import List, Optional, Literal

# --------- 資料模型 (原 /execute 用) ---------
class ChatMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    phone_number: str
    messages: List[ChatMessage]

# --------- 新的資料模型 (給後端工程師串接用) ---------
class HistoryMessage(BaseModel):
    """過去對話的單則訊息 (採用 LangChain 標準格式)"""
    type: Literal["human", "ai"]   # "human" = 客人 / "ai" = AI 或真人客服回覆
    content: str

class TreatmentFee(BaseModel):
    """療程費用單筆資料"""
    name: str                       # 療程名稱（含時長/組合等細節），例如 "NEO-熱磁減脂(30分鐘) + 冷凍單點(60分鐘)"
    price: int                      # 體驗價，純數字，例如 15999

class BackendUserQuery(BaseModel):
    fb_account: str                             # FB 使用者識別 (Messenger PSID)
    content: Optional[str] = None               # 客人這次傳的文字訊息（圖片訊息時為 None）
    image_url: List[str] = []                   # 客人這次傳的圖片 URL 陣列（文字訊息時為空 list；圖片訊息可能含多張）
    message_history: List[HistoryMessage] = []  # 過去對話（不含本次）
    ad_referral: Optional[str] = None           # 客人從 Meta 廣告進來的療程關鍵字，建議只在首次對話時傳
    treatment_fees: List[TreatmentFee] = []     # 目前療程費用表（每次都傳最新，後端可加 cache）

class BackendResponse(BaseModel):
    text: str               # 回覆文字（CallCS=1 時為空字串）
    images: List[str] = []  # 圖片 URL 列表（CallCS=1 或 2 時為空 list）
    CallCS: int = 0         # 0 = 正常 / 1 = 客人找真人客服（無 text、無 images）/ 2 = 預約流程（有 text、無 images，後端先發 text 再通知客服）