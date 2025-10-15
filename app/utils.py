import re

def is_valid_phone_number(text: str) -> bool:
    """檢查台灣手機號碼格式"""
    return bool(re.match(r"^09\d{8}$", text))