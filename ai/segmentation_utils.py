import requests
import numpy as np
from PIL import Image
import io

def segment_clothes(image: Image.Image, fallback_local=True):
    """
    Human parsing: แยกส่วนเสื้อ/กางเกง
    - ถ้าออนไลน์: ใช้ HuggingFace API (LIP)
    - ถ้าออฟไลน์หรือ API fail: ใช้ heuristic (แบ่งครึ่งบน=เสื้อ, ครึ่งล่าง=กางเกง เฉพาะ pixel ที่ไม่โปร่งใส)
    คืน mask (np.ndarray) ที่ label: 5=upper-clothes, 6=pants
    """
    # --- Try online API ---
    API_URL = "https://hf.space/embed/akhaliq/LIP_Human_Parsing/api/predict/"
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                mask_bytes = bytes(result["data"][0]["mask"]["data"])
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask = np.array(mask_img)
                return mask
        else:
            print(f"Online API error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Online API exception: {e}")
    # --- Fallback: local heuristic (offline, always returns mask) ---
    if fallback_local:
        arr = np.array(image)
        h, w = arr.shape[0], arr.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        # RGBA: use alpha > 0 as foreground
        if arr.shape[-1] == 4:
            alpha = arr[...,3]
            # Upper half = shirt, lower half = pants
            upper = (np.arange(h) < h//2)[:,None]
            mask[(alpha > 0) & upper] = 5
            mask[(alpha > 0) & (~upper)] = 6
        else:
            # RGB: assume all pixels are foreground
            mask[:h//2, :] = 5
            mask[h//2:, :] = 6
        # If mask is all zeros (empty), fallback to all foreground
        if np.all(mask == 0):
            mask[:h//2, :] = 5
            mask[h//2:, :] = 6
        return mask
    # If all fails, return None
    print("segment_clothes: No mask produced (offline fallback)")
    return None

def extract_part(image: Image.Image, mask: np.ndarray, part_labels):
    """
    คืนภาพเฉพาะส่วนที่ label อยู่ใน part_labels (list of int)
    """
    arr = np.array(image)
    part_mask = np.isin(mask, part_labels)
    if arr.shape[-1] == 3:
        arr_rgba = np.concatenate([arr, np.full(arr.shape[:2]+(1,), 255, dtype=np.uint8)], axis=-1)
    else:
        arr_rgba = arr.copy()
    arr_rgba[...,3] = np.where(part_mask, 255, 0)
    return Image.fromarray(arr_rgba, mode="RGBA")
