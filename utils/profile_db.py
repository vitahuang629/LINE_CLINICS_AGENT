from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()
clinic_user = os.getenv("CLINIC_USER")
clinic_pass = os.getenv("CLINIC_PASS")
clinic_host = os.getenv("CLINIC_HOST")
clinic_port = os.getenv("CLINIC_PORT")
clinic_db = os.getenv("CLINIC_DB")


CLINIC_USER = clinic_user
CLINIC_PASS = clinic_pass
CLINIC_HOST = clinic_host
CLINIC_PORT = clinic_port
CLINIC_DB = clinic_db

clinic_engine = create_engine(
    f'mysql+pymysql://{CLINIC_USER}:{CLINIC_PASS}@{CLINIC_HOST}:{CLINIC_PORT}/{CLINIC_DB}?charset=utf8'
)


def get_user_profile_by_uuid(line_uuid: str) -> dict:
    """
    根據 LINE UUID 查詢使用者的歷史對話記錄
    
    Args:
        line_uuid: LINE 使用者的 UUID
        
    Returns:
        dict: 包含歷史對話的字典，格式如下：
        {
            'line_name': '使用者暱稱',
            'conversation_history': [
                {'role': 'user', 'content': '訊息內容', 'timestamp': '時間'},
                ...
            ]
        }
        如果查無資料則回傳空字典
    """
    # 查詢最近的對話記錄
    query = text(f"""
        SELECT
            LA.line_id,
            LA.line_name,
            LM.message,
            LM.created_at
        FROM line_account LA
        LEFT JOIN line_message LM ON LA.id = LM.line_account_id
        WHERE LA.line_id = '{line_uuid}'
        ORDER BY LM.created_at DESC
        LIMIT 20;
    """)

    try:
        with clinic_engine.connect() as conn:
            resultDf = pd.read_sql(query, conn)
            
            # 如果查無資料，回傳空字典（表示新客戶）
            if resultDf.empty:
                print(f'⚠️ 查無資料: {line_uuid}（新客戶）')
                return {}
            
            # 將對話記錄整理成結構化格式
            conversation_history = []
            for _, row in resultDf.iterrows():
                if pd.notna(row['message']):
                    conversation_history.append({
                        'role': row['role'] if pd.notna(row['role']) else 'user',
                        'content': row['message'],
                        'timestamp': str(row['created_at']) if pd.notna(row['created_at']) else ''
                    })
            
            # 反轉順序（因為查詢是 DESC，需要改成時間正序）
            conversation_history.reverse()
            
            result = {
                'line_name': resultDf['line_name'].iloc[0] if pd.notna(resultDf['line_name'].iloc[0]) else '',
                'conversation_history': conversation_history
            }
            
            print(f'✅ 找到使用者資料: {line_uuid}, 對話記錄數: {len(conversation_history)}')
            return result

    except Exception as e:
        print(f"❌ Error in get_user_profile_by_uuid: {e}")
        return {}


def get_user_profile_by_phone(identifier: str) -> dict:
    """
    根據識別碼查詢使用者資料
    現在統一使用 LINE UUID，此函數保留作為相容性介面
    
    Args:
        identifier: LINE UUID 或手機號碼
        
    Returns:
        dict: 使用者的歷史對話記錄，如果查無資料則回傳空字典
    """
    # 直接呼叫 UUID 查詢函數
    return get_user_profile_by_uuid(identifier)
