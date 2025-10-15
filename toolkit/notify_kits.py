import pandas as pd
from typing import Literal, Dict, Any
from langchain_core.tools import tool
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel
import os
from utils.ensemble_retriever import get_ensemble_retriever
import random

print("notify_kits.py is running")


def check_appointment_keywords(response_text: str) -> bool:
    """
    檢查 AI 回覆中是否包含預約相關的關鍵字
    """
    appointment_keywords = [
        "送出", "紀錄", "稍後", "盡快", "專人"
    ]
    
    # 計算關鍵字出現次數
    keyword_count = 0
    for keyword in appointment_keywords:
        if keyword in response_text:
            keyword_count += 1
    
    # 如果有足夠的關鍵字，認為是預約回覆
    return keyword_count >= 2

# def notify_human_agent(user_id: str, appointment_info: Dict = None):
#     """
#     通知真人客服接手處理
#     """ 
#     notification_message = (
#         f"🔔 需要真人客服接手！\n"
#         f"用戶 ID: {user_id}\n"
#         f"預約資訊: {appointment_info if appointment_info else '無'}"
#     )
    
#     print(f"通知客服: {notification_message}")
#     return notification_message