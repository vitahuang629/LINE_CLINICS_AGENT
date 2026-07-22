from sqlalchemy import create_engine, text
import json
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

def extract_name_active_or_value_dict(row):
    try:
        items = json.loads(row)
        if not isinstance(items, list):
            return row

        result_dict = {}
        for item in items:
            name = item.get("name")
            active = item.get("active")
            value = item.get("value")

            final_value = active if active is not None else value

            # 👉 加這段：如果是 None、空字串或空 list，就跳過
            if final_value is None or (isinstance(final_value, str) and final_value.strip() == ""):
                continue
            if isinstance(final_value, list) and not final_value:
                continue

            # 處理成列表形式
            if isinstance(final_value, list):
                final_value_str = final_value
            else:
                final_value_str = [str(final_value).strip()]

            if name is None or str(name).strip() == "":
                result_dict.setdefault("unnamed", []).extend(final_value_str)
            else:
                result_dict[str(name).strip()] = final_value_str

        return result_dict

    except Exception:
        return row

def get_user_profile_by_phone(mobilephone: str) -> dict:
    query = text(f"""
             SELECT
                M.member_birthday,
                CASE
                    WHEN M.member_sex = 1 THEN '男'
                    ELSE '女'
                END AS member_sex,
                DC.comment
            FROM member M
            LEFT JOIN doctor_comments DC
            ON M.id = DC.member_id 
            WHERE M.member_mobile  = '{mobilephone}' and comment IS NOT NULL
            ORDER BY M.created_at DESC 
            LIMIT 1;
""")

    try:
        with clinic_engine.connect() as conn:
            resultDf = pd.read_sql(query, conn)
            resultDf['comment_final'] = resultDf['comment'].apply(extract_name_active_or_value_dict)
            print('userrrrrrrrr', resultDf['comment_final'].iloc[0])
            
            # 如果只有一筆結果，就回傳第一筆
            if not resultDf.empty:
                print('難道是這裡')
                return resultDf['comment_final'].iloc[0]  # 通常給 state["user_profile"]
            else:
                print('跑來這裡嗎')
                return {}

    except Exception as e:
        print(f"❌ Error in get_user_profile_by_phone: {e}")
        return {}