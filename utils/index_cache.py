"""索引快取的新鮮度控管。

Chroma 與 BM25 的建立成本不對稱：BM25 只是 jieba 斷詞，重建幾乎免費；
Chroma 要把每個 chunk 送去 OpenAI embedding，是真的會花時間跟錢的那一半。
所以索引值得快取，但「快取存在」不等於「快取是對的」——
舊版只判斷 persist_dir 在不在，會踩到兩種情況：

1. bind mount 首次掛載時 Docker 會建一個空目錄，目錄「存在」但裡面沒東西，
   於是載入到一個空 collection，檢索永遠回 0 筆而且完全不報錯。
2. data/*.csv 更新後目錄照樣存在，舊索引就這樣上線。

這裡改用「來源指紋」當守門條件：指紋相符才用快取，否則重建。
"""

import hashlib
import os
import shutil

# 指紋檔名（放在 persist_dir 內；BM25 則接在 pickle 檔名後面）
MARKER_NAME = ".source_fingerprint"


def source_fingerprint(csv_path: str, logic_version: str) -> str:
    """以「來源 CSV 內容 + 建索引邏輯版本」算出指紋。

    logic_version 是給人手動 bump 的：CSV 沒動但你改了 Document 的組法
    （例如把 category 拼進 page_content、調整加權重複次數），舊索引一樣是過期的，
    這時把呼叫端的版本字串 +1 就會強制重建。
    """
    h = hashlib.sha256()
    h.update(logic_version.encode("utf-8"))
    h.update(b"\0")
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chroma_marker_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, MARKER_NAME)


def bm25_marker_path(bm25_path: str) -> str:
    return f"{bm25_path}{MARKER_NAME}"


def is_fresh(marker_path: str, fingerprint: str) -> bool:
    """快取是否可用。指紋檔不存在（含空目錄）或內容不符都視為不新鮮。"""
    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return f.read().strip() == fingerprint
    except OSError:
        return False


def write_marker(marker_path: str, fingerprint: str) -> None:
    """寫入指紋。務必在索引「建立成功之後」才呼叫，
    否則建到一半掛掉會留下一個假的有效標記，下次啟動就直接載入半殘的索引。
    """
    parent = os.path.dirname(marker_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(fingerprint)


def clear_dir_contents(path: str) -> None:
    """清空目錄內容，但保留目錄本身。

    不能用 shutil.rmtree(path)：這個目錄很可能是 bind mount 的掛載點，
    刪掉掛載點本身在 Linux 上會噴 OSError: Device or resource busy。
    """
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full, ignore_errors=True)
        else:
            try:
                os.unlink(full)
            except OSError:
                pass
