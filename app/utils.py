import re

# ⚠️ 目前無人呼叫（原使用者 app/line_service_*.py 已刪除），但**刻意保留**：
#    之後要做「直接幫客人完成預約」時，收到的手機號碼需要這個格式驗證。
#    做 dead code 清理時請跳過這個檔案。


def is_valid_phone_number(text: str) -> bool:
    """檢查台灣手機號碼格式"""
    return bool(re.match(r"^09\d{8}$", text))