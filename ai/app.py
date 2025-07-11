import streamlit as st
import numpy as np
from PIL import Image, ImageColor
import requests
from segmentation_utils import segment_clothes, extract_part

# --- ติดตั้ง scikit-learn, webcolors, backgroundremover, opencv-python อัตโนมัติถ้าไม่มี ---
try:
    from sklearn.cluster import KMeans
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.cluster import KMeans
try:
    import webcolors
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'webcolors'])
    import webcolors
try:
    from backgroundremover import remove as remove_bg_sota
    BGREMOVER_AVAILABLE = True
except ImportError:
    BGREMOVER_AVAILABLE = False
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
    import cv2
    CV2_AVAILABLE = True
import io

# ตารางสีมาตรฐาน (ไทย, อังกฤษ, RGB, HEX)
COLOR_TABLE = [
    # โทนร้อน
    {"th": "แดงสด", "en": "Crimson Red", "rgb": (220, 20, 60), "hex": "#DC143C"},
    {"th": "ส้ม", "en": "Dark Orange", "rgb": (255, 140, 0), "hex": "#FF8C00"},
    {"th": "เหลือง", "en": "Gold Yellow", "rgb": (255, 215, 0), "hex": "#FFD700"},
    {"th": "ทอง", "en": "Goldenrod", "rgb": (218, 165, 32), "hex": "#DAA520"},
    # โทนเย็น
    {"th": "ฟ้าน้ำทะเล", "en": "Deep Sky Blue", "rgb": (0, 191, 255), "hex": "#00BFFF"},
    {"th": "ฟ้าอ่อน", "en": "Light Blue", "rgb": (173, 216, 230), "hex": "#ADD8E6"},
    {"th": "น้ำเงิน", "en": "Medium Blue", "rgb": (0, 0, 205), "hex": "#0000CD"},
    {"th": "ม่วง", "en": "Blue Violet", "rgb": (138, 43, 226), "hex": "#8A2BE2"},
    {"th": "เขียวมะนาว", "en": "Lime Green", "rgb": (50, 205, 50), "hex": "#32CD32"},
    {"th": "เขียวเข้ม", "en": "Dark Green", "rgb": (0, 100, 0), "hex": "#006400"},
    # พาสเทล
    {"th": "ชมพูพาสเทล", "en": "Pastel Pink", "rgb": (255, 182, 193), "hex": "#FFB6C1"},
    {"th": "เหลืองพาสเทล", "en": "Pastel Yellow", "rgb": (255, 255, 153), "hex": "#FFFF99"},
    {"th": "เขียวพาสเทล", "en": "Pastel Green", "rgb": (144, 238, 144), "hex": "#90EE90"},
    {"th": "ฟ้าพาสเทล", "en": "Pastel Blue", "rgb": (173, 216, 230), "hex": "#ADD8E6"},
    {"th": "ม่วงพาสเทล", "en": "Pastel Purple", "rgb": (216, 191, 216), "hex": "#D8BFD8"},
    # คลาสสิก
    {"th": "ดำ", "en": "Black", "rgb": (0, 0, 0), "hex": "#000000"},
    {"th": "ขาว", "en": "White", "rgb": (255, 255, 255), "hex": "#FFFFFF"},
   # {"th": "เทา", "en": "Gray", "rgb": (128, 128, 128), "hex": "#808080"},
    {"th": "น้ำตาล", "en": "Brown", "rgb": (139, 69, 19), "hex": "#8B4513"},
    # โทนธรรมชาติ (Earth Tone)
    {"th": "น้ำตาลอ่อน", "en": "Peru", "rgb": (205, 133, 63), "hex": "#CD853F"},
    {"th": "เขียวมะกอก", "en": "Olive Drab", "rgb": (107, 142, 35), "hex": "#6B8E23"},
    {"th": "ทราย", "en": "Sandy Brown", "rgb": (244, 164, 96), "hex": "#F4A460"},
    {"th": "ครีม", "en": "Cream", "rgb": (255, 253, 208), "hex": "#FFFDD0"},
    # สีมาตรฐานเดิม (บางส่วน)
    {"th": "แดง", "en": "Red", "rgb": (255, 0, 0), "hex": "#FF0000"},
    {"th": "เขียว", "en": "Green", "rgb": (0, 128, 0), "hex": "#008000"},
    {"th": "น้ำเงิน", "en": "Blue", "rgb": (0, 0, 255), "hex": "#0000FF"},
    {"th": "เหลือง", "en": "Yellow", "rgb": (255, 255, 0), "hex": "#FFFF00"},
    {"th": "ชมพู", "en": "Pink", "rgb": (255, 192, 203), "hex": "#FFC0CB"},
    {"th": "ส้ม", "en": "Orange", "rgb": (255, 165, 0), "hex": "#FFA500"},
    {"th": "ม่วง", "en": "Purple", "rgb": (128, 0, 128), "hex": "#800080"},
    {"th": "ฟ้าอ่อน", "en": "Light Blue", "rgb": (173, 216, 230), "hex": "#ADD8E6"},
    {"th": "เทาอ่อน", "en": "Light Gray", "rgb": (211, 211, 211), "hex": "#D3D3D3"},
   # {"th": "เทาเข้ม", "en": "Dark Gray", "rgb": (105, 105, 105), "hex": "#696969"},
    {"th": "ครีม / เบจ", "en": "Beige", "rgb": (245, 245, 220), "hex": "#F5F5DC"},
    {"th": "เขียวมะกอก", "en": "Olive", "rgb": (128, 128, 0), "hex": "#808000"},
    {"th": "แดงเลือดหมู", "en": "Maroon", "rgb": (128, 0, 0), "hex": "#800000"},
    {"th": "ฟ้าเข้ม", "en": "Navy", "rgb": (0, 0, 128), "hex": "#000080"},
    {"th": "ฟ้าเทอร์ควอยซ์", "en": "Turquoise", "rgb": (64, 224, 208), "hex": "#40E0D0"},
    {"th": "ทอง", "en": "Gold", "rgb": (255, 215, 0), "hex": "#FFD700"},
    {"th": "เงิน", "en": "Silver", "rgb": (192, 192, 192), "hex": "#C0C0C0"},
]

# ---------------- ฟังก์ชันวิเคราะห์สี ----------------
def get_dominant_colors(image, k=10):
    """
    คืน dominant color (RGB) จากภาพ โดยไม่ดูดสีพื้นหลัง (pixel โปร่งใส, ขาว, เทา)
    - ใช้ k-means เฉพาะ pixel ที่ alpha > 0 (ถ้ามี alpha)
    - กรองขาว/เทาออก
    - ถ้า pixel เหลือน้อย fallback ใช้ pixel ทั้งหมด
    """
    arr = np.array(image)
    h, w = arr.shape[0], arr.shape[1]
    # ถ้าเป็น RGBA: ใช้ alpha > 0 เป็น foreground mask
    if arr.shape[-1] == 4:
        alpha = arr[...,3]
        mask_fg = alpha > 0
    else:
        mask_fg = np.ones((h, w), dtype=bool)
    # ใช้ connected component เพื่อหา region ที่ใหญ่สุด (คน/เสื้อผ้า)
    try:
        import cv2
        mask_fg_uint8 = (mask_fg*255).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(mask_fg_uint8)
        # กรอง region เล็ก ๆ ออก (noise)
        label_counts = np.bincount(labels_im.flatten())
        label_counts[0] = 0
        main_label = np.argmax(label_counts)
        main_mask = labels_im == main_label
        arr_fg = arr[...,:3][main_mask]
    except Exception:
        arr_fg = arr[...,:3][mask_fg]
    # กรอง pixel ขาว/เทา (background) แบบละเอียดขึ้น
    mask = ~(
        ((arr_fg[:,0]>220) & (arr_fg[:,1]>220) & (arr_fg[:,2]>220)) |  # ขาว
        ((np.abs(arr_fg[:,0]-arr_fg[:,1])<15) & (np.abs(arr_fg[:,1]-arr_fg[:,2])<15) & (arr_fg[:,0]>80) & (arr_fg[:,0]<210)) # เทา
    )
    arr_fg = arr_fg[mask]
    # กรอง outlier สีด้วย median filter (ลดผลกระทบจาก pixel noise)
    if len(arr_fg) > 0:
        med = np.median(arr_fg, axis=0)
        arr_fg = arr_fg[np.linalg.norm(arr_fg-med, axis=1)<80]
    # ถ้า pixel foreground เหลือน้อย fallback ใช้ pixel ทั้งหมด
    if len(arr_fg) < k:
        arr_fg = arr.reshape(-1,3)
    # ลดจำนวน pixel เพื่อความเร็ว
    if len(arr_fg) > 20000:
        idx = np.random.choice(len(arr_fg), 20000, replace=False)
        arr_fg = arr_fg[idx]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=5).fit(arr_fg)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = kmeans.cluster_centers_[sorted_idx].astype(int)
    return colors

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(tuple(rgb_tuple))
    except ValueError:
        min_diff = float('inf')
        closest_name = ''
        # รองรับ webcolors หลายเวอร์ชัน
        if hasattr(webcolors, 'CSS3_NAMES'):
            color_names = webcolors.CSS3_NAMES
        elif hasattr(webcolors, 'CSS3_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'HTML4_NAMES_TO_HEX'):
            color_names = list(webcolors.HTML4_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'CSS21_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS21_NAMES_TO_HEX.keys())
        else:
            color_names = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray']
        for name in color_names:
            try:
                r_c, g_c, b_c = webcolors.name_to_rgb(name)
                diff = np.linalg.norm(np.array([r_c, g_c, b_c]) - np.array(rgb_tuple))
                if diff < min_diff:
                    min_diff = diff
                    closest_name = name
            except Exception:
                continue
        return closest_name

def get_color_name_th(rgb_tuple):
    # เทียบกับ COLOR_TABLE ก่อน (ใช้ Euclidean distance)
    min_dist = float('inf')
    best = None
    for c in COLOR_TABLE:
        dist = sum((a-b)**2 for a, b in zip(rgb_tuple, c['rgb']))
        if dist < min_dist:
            min_dist = dist
            best = c
    if min_dist < 900:  # ถ้าใกล้เคียงมากพอ (เช่น ห่างไม่เกิน ~30 ต่อ channel)
        return best['th']
    # fallback เดิม
    color_th = {
        'black': 'ดำ', 'white': 'ขาว', 'red': 'แดง', 'green': 'เขียว', 'blue': 'น้ำเงิน',
        'yellow': 'เหลือง', 'cyan': 'ฟ้าอมเขียว', 'magenta': 'ชมพู', 'gray': 'เทา',
        'orange': 'ส้ม', 'pink': 'ชมพู', 'purple': 'ม่วง', 'brown': 'น้ำตาล',
        'gold': 'ทอง', 'silver': 'เงิน', 'beige': 'เบจ', 'navy': 'กรมท่า',
        'maroon': 'แดงเลือดหมู', 'olive': 'เขียวมะกอก', 'teal': 'เขียวอมฟ้า',
        'lime': 'เขียวอ่อน', 'indigo': 'คราม', 'violet': 'ม่วงอ่อน',
        'turquoise': 'ฟ้าอมเขียว', 'coral': 'ส้มอมชมพู', 'salmon': 'ส้มอมชมพู',
        'khaki': 'กากี', 'lavender': 'ม่วงลาเวนเดอร์', 'skyblue': 'ฟ้า',
        'aqua': 'ฟ้า', 'azure': 'ฟ้าอ่อน', 'ivory': 'ขาวงาช้าง', 'tan': 'น้ำตาลอ่อน',
        'chocolate': 'น้ำตาลเข้ม', 'plum': 'ม่วงพลัม', 'orchid': 'ม่วงกล้วยไม้',
        'crimson': 'แดงเข้ม', 'tomato': 'แดงอมส้ม', 'peachpuff': 'พีช',
        'mintcream': 'เขียวมิ้นท์', 'aliceblue': 'ฟ้าอ่อน', 'slategray': 'เทาเข้ม',
        'lightgray': 'เทาอ่อน', 'darkgray': 'เทาเข้ม', 'darkblue': 'น้ำเงินเข้ม',
        'lightblue': 'ฟ้าอ่อน', 'darkgreen': 'เขียวเข้ม', 'lightgreen': 'เขียวอ่อน',
        'darkred': 'แดงเข้ม', 'lightpink': 'ชมพูอ่อน', 'darkorange': 'ส้มเข้ม',
        'goldenrod': 'ทองเข้ม', 'firebrick': 'แดงอิฐ', 'sienna': 'น้ำตาลแดง',
        'rosybrown': 'น้ำตาลอมชมพู', 'peru': 'น้ำตาลทอง', 'wheat': 'ข้าวสาลี',
        'seashell': 'ขาวเปลือกหอย', 'linen': 'ขาวลินิน', 'oldlace': 'ขาวลูกไม้',
        'snow': 'ขาวหิมะ', 'honeydew': 'ขาวอมเขียว', 'floralwhite': 'ขาวอมเหลือง',
        'ghostwhite': 'ขาวอมฟ้า', 'whitesmoke': 'ขาวหมอก', 'gainsboro': 'เทาอ่อน',
        'mediumblue': 'น้ำเงินกลาง', 'mediumseagreen': 'เขียวกลาง',
        'mediumvioletred': 'ชมพูม่วง', 'mediumorchid': 'ม่วงกลาง',
        'mediumslateblue': 'น้ำเงินม่วง', 'mediumturquoise': 'ฟ้าอมเขียวกลาง',
        'mediumspringgreen': 'เขียวอ่อนสด', 'mediumaquamarine': 'ฟ้าอมเขียวอ่อน',
        'mediumpurple': 'ม่วงกลาง', 'midnightblue': 'น้ำเงินเข้มมาก',
        'lightyellow': 'เหลืองอ่อน', 'lightgoldenrodyellow': 'เหลืองทองอ่อน',
        'lightcoral': 'ส้มอมชมพูอ่อน', 'lightcyan': 'ฟ้าอมเขียวอ่อน',
        'lightseagreen': 'เขียวอมฟ้าอ่อน', 'lightsalmon': 'ส้มอมชมพูอ่อน',
        'lightsteelblue': 'ฟ้าอมเทา', 'lightgray': 'เทาอ่อน',
        'darkslategray': 'เทาเข้ม', 'darkolivegreen': 'เขียวมะกอกเข้ม',
        'darkmagenta': 'ม่วงเข้ม', 'darkviolet': 'ม่วงเข้ม',
        'darkorchid': 'ม่วงเข้ม', 'darkgoldenrod': 'ทองเข้ม',
        'darkkhaki': 'กากีเข้ม', 'darkseagreen': 'เขียวอ่อนเข้ม',
        'darkturquoise': 'ฟ้าอมเขียวเข้ม', 'darkcyan': 'ฟ้าอมเขียวเข้ม',
        'darkslateblue': 'น้ำเงินม่วงเข้ม', 'darkorange': 'ส้มเข้ม',
        'darkred': 'แดงเข้ม', 'darksalmon': 'ส้มอมชมพูเข้ม',
        'darkgray': 'เทาเข้ม', 'darkblue': 'น้ำเงินเข้ม',
        'darkgreen': 'เขียวเข้ม', 'darkslategray': 'เทาเข้ม',
        'darkviolet': 'ม่วงเข้ม', 'darkorchid': 'ม่วงเข้ม',
        'darkgoldenrod': 'ทองเข้ม', 'darkkhaki': 'กากีเข้ม',
        'darkseagreen': 'เขียวอ่อนเข้ม', 'darkturquoise': 'ฟ้าอมเขียวเข้ม',
        'darkcyan': 'ฟ้าอมเขียวเข้ม', 'darkslateblue': 'น้ำเงินม่วงเข้ม',
        'rebeccapurple': 'ม่วงรีเบคก้า',
    }
    en = get_color_name(rgb_tuple)
    return color_th.get(en.lower(), en)

# ---------------- ฟังก์ชันวิเคราะห์ความเข้ากันของสี ----------------
def evaluate_color_match(colors):
    scores = []
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            diff = np.linalg.norm(colors[i] - colors[j])
            scores.append(diff)
    avg_diff = np.mean(scores)
    std_diff = np.std(scores)
    min_diff = np.min(scores)
    max_diff = np.max(scores)
    # ปรับช่วงคะแนนให้ยืดหยุ่นขึ้น (40-90) และคำแนะนำละเอียดขึ้น
    if avg_diff < 30:
        return "โทนสีใกล้เคียงกันมาก เหมาะกับแนวมินิมอล/สุภาพ/เรียบหรู ดูสบายตา", 90
    elif avg_diff < 45:
        return "สีใกล้เคียงกัน เหมาะกับแนว casual, minimal, business casual, everyday look", 85
    elif avg_diff < 60:
        if std_diff < 15:
            return "สีหลักและสีรองกลมกลืน เหมาะกับแนว smart casual, soft tone", 80
        else:
            return "มีสีหลักโดดเด่น สีรองช่วยเสริม เหมาะกับแนวแฟชั่น/โมเดิร์น", 75
    elif avg_diff < 75:
        if std_diff > 30:
            return "สีตัดกันพอดี ดูมีสไตล์ เหมาะกับแนว creative, modern, street", 70
        else:
            return "สีตัดกันเล็กน้อย เพิ่มความน่าสนใจ เหมาะกับแนว everyday, pop", 68
    elif avg_diff < 90:
        if max_diff > 150:
            return "สีตัดกันชัดเจน เหมาะกับแนวแฟชั่นจัดจ้าน/สายฝอ/ปาร์ตี้/experimental", 60
        else:
            return "สีตัดกันแรงแต่ยังดูดี เหมาะกับแนวแฟชั่น/creative/statement look", 55
    else:
        return "สีตัดกันแรงมาก อาจดูขัดตา เหมาะกับลุคแฟชั่นจัดเต็มหรือ experimental (ควรเลือกสีรองใหม่)", 40

# ---------------- ฟังก์ชันทำนายแนวแต่งตัว ----------------
def predict_style(colors):
    # วิเคราะห์จากสีหลักและสีรอง
    main_color = colors[0]
    if len(colors) > 1:
        second_color = colors[1]
    else:
        second_color = main_color
    r, g, b = main_color
    r2, g2, b2 = second_color
    # เงื่อนไขซับซ้อนขึ้นและเพิ่มสไตล์ใหม่ ๆ
    if r < 80 and g < 80 and b < 80:
        return "แนวเท่ (Street / ดาร์กแฟชั่น) - โทนเข้ม/ดำ/เทา"
    elif r > 200 and g > 200 and b > 200:
        return "แนวหวาน (Pastel / ญี่ปุ่น) - โทนขาว/พาสเทล"
    elif r > 200 and g < 100 and b < 100:
        if abs(r2 - r) > 80 or abs(g2 - g) > 80 or abs(b2 - b) > 80:
            return "แนวแฟชั่นจัดจ้าน (สายฝอ/Pop) - สีสดตัดกัน"
        else:
            return "แนวแฟชั่นสดใส (Pop/Colorful)"
    elif abs(r-g) < 30 and abs(g-b) < 30 and abs(r-b) < 30 and r > 150:
        return "แนวสุภาพ/เรียบหรู (Smart Casual/Minimal) - โทนสีเดียวกัน"
    elif (r > 180 and g > 180) or (g > 180 and b > 180) or (r > 180 and b > 180):
        return "แนวหวาน/สดใส (Pastel/ญี่ปุ่น/เกาหลี)"
    elif max(r, g, b) - min(r, g, b) > 150:
        if r > 200 and g > 200 and b < 100:
            return "แนว Summer สดใส (เหลือง/ส้ม/ฟ้า)"
        elif r < 100 and g > 150 and b < 100:
            return "แนว Earth Tone (เขียว/น้ำตาล/ธรรมชาติ)"
        elif r > 200 and g < 100 and b > 200:
            return "แนว Neon/Retro (ชมพู/ม่วง/ฟ้า)"
        elif r < 100 and g < 100 and b > 180:
            return "แนว Denim/Blue Jeans (น้ำเงิน/ฟ้า)"
        elif r > 180 and g > 120 and b < 80:
            return "แนว Autumn (น้ำตาล/ส้ม/เหลือง)"
        elif r < 100 and g > 100 and b > 100:
            return "แนว Winter (ฟ้า/เขียว/ขาว)"
        elif r > 200 and g > 200 and b > 200:
            return "แนว Spring (พาสเทล/สดใส)"
        elif r > 180 and g > 180 and b > 180:
            return "แนว Monochrome (ขาว/เทา/ดำ)"
        else:
            return "แนวแฟชั่น/สตรีท/Creative - สีตัดกันชัด"
    elif r > 150 and g > 150 and b < 100:
        return "แนว Luxury/Business (ทอง/เหลือง/น้ำตาล)"
    elif r < 100 and g > 150 and b > 150:
        return "แนว Sport/Active (ฟ้า/เขียว/น้ำเงิน)"
    else:
        return "แนวมินิมอล / สุภาพ / Everyday Look"

def remove_background(image):
    """
    ลบพื้นหลัง: ใช้ backgroundremover (ดีที่สุด) ถ้ามี, fallback เป็น rembg, ถ้าไม่มีคืนภาพเดิม
    คืน PIL.Image RGBA
    """
    if BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove_bg_sota(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    if REMBG_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    return image.convert("RGBA")

def remove_background_bytes(image_bytes):
    """ลบพื้นหลังจาก bytes (input: bytes, output: PIL Image RGBA หรือ None) พร้อม refine ขอบให้เนียน"""
    if not REMBG_AVAILABLE:
        return None
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        output_image = remove(input_image)
        output_image = output_image.convert("RGBA")
        # refine ขอบ alpha
        output_image = refine_alpha_edges(output_image, method="morph+blur", ksize=5, blur_sigma=1.2)
        return output_image
    except Exception:
        return None

def remove_background_bytes_v2(image_bytes):
    """
    ลบพื้นหลังจาก bytes (input: bytes, output: PIL Image RGBA หรือ None)
    ใช้ rembg แบบตรงไปตรงมา (ไม่ refine, minimal logic)
    """
    if not REMBG_AVAILABLE:
        return None
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        output_image = remove(input_image)
        return output_image.convert("RGBA")
    except Exception as e:
        print("rembg (bytes) error:", e)
        return None

def remove_background_bytes_v3(image_bytes):
    """
    ลบพื้นหลังด้วย rembg แบบ bytes-in bytes-out (robust ที่สุด)
    input: bytes (PNG/JPEG/WebP), output: PIL Image RGBA หรือ None
    """
    if not REMBG_AVAILABLE:
        return None
    try:
        from rembg import remove
        output_bytes = remove(image_bytes)  # bytes in, bytes out
        image_nobg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        return image_nobg
    except Exception as e:
        print("rembg (bytes-in, bytes-out) error:", e)
        return None

def manual_remove_bg(image, bg_color, tolerance=30):
    """
    ลบพื้นหลังโดยใช้ threshold สี (bg_color: hex หรือ tuple, tolerance: int)
    คืนค่า RGBA (พื้นหลังโปร่งใส)
    """
    img = image.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[0], arr.shape[1]
    # Adaptive thresholding: คำนวณ mean/std ของขอบภาพ
    # Even less aggressive: further increase border width
    border_width = max(10, int(min(h, w)*0.22))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = arr[border_mask][...,:3]
    mean_border = np.mean(border_pixels, axis=0)
    std_border = np.std(border_pixels, axis=0)
    # Set tolerance to 50 for moderate mask
    color_dist = np.linalg.norm(arr[...,:3] - mean_border, axis=-1)
    adaptive_mask = (color_dist < (std_border.mean() + 50)) & border_mask
    try:
        import cv2
        arr_bgr = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2BGR)
        # Initial mask for GrabCut: probable background (border), probable foreground (center)
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        mask[border_mask] = cv2.GC_BGD
        center_mask = np.zeros((h, w), dtype=bool)
        # Shrink center mask even more for less aggressive cut
        center_margin = int(min(h, w)*0.14)
        center_mask[center_margin:-center_margin, center_margin:-center_margin] = True
        mask[center_mask] = cv2.GC_PR_FGD
        # Run GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(arr_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        grabcut_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        # Edge detection: หาขอบวัตถุ
        arr_gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(arr_gray, 80, 180)
        # Even lower edge dilation for softer cut
        kernel = np.ones((2,2), np.uint8)
        edge_mask_dil = cv2.dilate(edges, kernel, iterations=0) > 0
        # Combine: keep foreground, avoid cutting into edge
        final_mask = grabcut_mask & (~edge_mask_dil)
        # Morphological closing for smooth mask
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        # Even lower blur for softer feathering
        mask_blur = cv2.GaussianBlur(final_mask.astype(np.float32), (1,1), 0.3)
        arr[...,3] = (mask_blur*255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        # Fallback: original adaptive mask + edge exclusion
        try:
            import cv2
            arr_gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(arr_gray, 80, 180)
            edge_mask = edges > 0
            kernel = np.ones((5,5), np.uint8)
            edge_mask_dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
            adaptive_mask2 = adaptive_mask & (~edge_mask_dil)
            adaptive_mask2 = cv2.morphologyEx(adaptive_mask2.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            arr[...,3][adaptive_mask2] = 0
            return Image.fromarray(arr)
        except Exception:
            arr[...,3][adaptive_mask] = 0
            return Image.fromarray(arr)

def call_hf_fashion_classifier(image: Image.Image):
    """
    ส่งภาพไปยัง HuggingFace Spaces Fashion-Classifier API และคืนค่าผลลัพธ์
    """
    API_URL = "https://hf.space/embed/KP-whatever/Fashion-Classifier/api/predict/"
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            # ปรับตามโครงสร้าง JSON ที่ได้จาก API
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]
            return str(result)
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"API Error: {e}"

def advanced_predict_style(colors, image=None):
    """
    วิเคราะห์สไตล์แฟชั่นโดยใช้ตรรกะที่ซับซ้อนขึ้น เช่น
    - วิเคราะห์ distribution ของสีทั้งภาพ (ไม่ใช่แค่ 2 สี)
    - ตรวจจับความสดใส/ความหม่น/ความ contrast
    - ใช้ข้อมูล saturation, brightness, และความหลากหลายของสี
    - คืนค่าความมั่นใจ (probability) ร่วมด้วย
    """
    import colorsys
    arr = np.array(image.resize((200,200))) if image is not None else None
    if arr is not None:
        arr = arr.reshape(-1,3)
        hsv = np.array([colorsys.rgb_to_hsv(*(pix/255.0)) for pix in arr])
        mean_sat = np.mean(hsv[:,1])
        mean_val = np.mean(hsv[:,2])
        std_val = np.std(hsv[:,2])
        color_variety = len(np.unique(arr, axis=0))
    else:
        mean_sat = mean_val = std_val = color_variety = 0
    main_color = colors[0]
    second_color = colors[1] if len(colors) > 1 else main_color
    r, g, b = main_color
    r2, g2, b2 = second_color
    # ตรรกะใหม่: วิเคราะห์ความสดใส, contrast, ความหลากหลายของสี, ฯลฯ
    if mean_val > 0.8 and mean_sat < 0.25:
        return ("แนวสุภาพ/เรียบหรู (Minimal/Smart Casual)", 95)
    elif mean_sat > 0.6 and color_variety > 10000:
        return ("แนวแฟชั่นสดใส/Pop/Colorful", 92)
    elif mean_val < 0.3 and std_val < 0.1:
        return ("แนวเท่/ดาร์กแฟชั่น (Street/Dark)", 90)
    elif mean_sat < 0.2 and std_val < 0.15:
        return ("แนว Monochrome/Classic", 88)
    elif abs(r-g)<20 and abs(g-b)<20 and mean_sat<0.3:
        return ("แนว Everyday Look / Casual", 85)
    elif mean_sat > 0.5 and mean_val > 0.5:
        return ("แนว Summer/สดใส/สายฝอ", 87)
    elif mean_sat < 0.3 and mean_val < 0.5:
        return ("แนว Earth Tone/Autumn", 83)
    elif color_variety < 2000:
        return ("แนว Minimal/เรียบง่าย", 80)
    else:
        return ("แนวแฟชั่น/Creative/Experimental", 75)

def refine_alpha_edges(image_rgba, method="morph+blur", ksize=3, blur_sigma=1.0):
    """
    ปรับขอบ alpha channel ให้คมขึ้น (morphological + blur)
    image_rgba: PIL Image RGBA
    method: "morph+blur" (default), "sharpen"
    คืนค่า PIL Image RGBA
    ถ้าไม่มี cv2 จะคืนภาพเดิม
    """
    try:
        import cv2
    except ImportError:
        # ถ้าไม่มี cv2 ให้คืนภาพเดิม ไม่ error
        return image_rgba
    arr = np.array(image_rgba)
    alpha = arr[...,3]
    kernel = np.ones((ksize,ksize), np.uint8)
    alpha_morph = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha_blur = cv2.GaussianBlur(alpha_morph, (ksize|1,ksize|1), blur_sigma)
    if method == "sharpen":
        sharp = cv2.addWeighted(alpha_blur, 1.5, cv2.GaussianBlur(alpha_blur, (0,0), 2), -0.5, 0)
        alpha_final = np.clip(sharp, 0, 255).astype(np.uint8)
    else:
        alpha_final = alpha_blur
    arr[...,3] = alpha_final
    return Image.fromarray(arr)

def checkerboard_bg(img, size=8):
    """
    วาดลายหมากรุก checkerboard เป็นพื้นหลังให้ภาพโปร่งใส (PIL RGBA)
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    bg = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(0, h, size):
        for x in range(0, w, size):
            color = 220 if (x//size + y//size) % 2 == 0 else 180
            bg[y:y+size, x:x+size, :3] = color
            bg[y:y+size, x:x+size, 3] = 255
    out = arr.copy()
    alpha = arr[...,3:4]/255.0
    out = (arr[...,:3]*alpha + bg[...,:3]*(1-alpha)).astype(np.uint8)
    out = np.concatenate([out, np.full((h,w,1),255,dtype=np.uint8)], axis=-1)
    return Image.fromarray(out)

def remove_background_modnet_api(image):
    """
    ลบพื้นหลังด้วย MODNet (HuggingFace API)
    คืน PIL.Image RGBA หรือ None
    """
    API_URL = "https://hf.space/embed/andreasjansson/modnet/api/predict/"
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                from base64 import b64decode
                img_bytes = b64decode(result["data"][0].split(",")[-1])
                return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        return None
    except Exception:
        return None

def remove_background_all(image):
    """
    ลบพื้นหลังทุกวิธี (backgroundremover, rembg)
    คืน dict {method: PIL.Image RGBA}
    """
    results = {}
    # backgroundremover
    if BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove_bg_sota(img_rgba)
            results['backgroundremover'] = img_no_bg.convert("RGBA")
        except Exception:
            results['backgroundremover'] = None
    # rembg
    if REMBG_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove(img_rgba)
            results['rembg'] = img_no_bg.convert("RGBA")
        except Exception:
            results['rembg'] = None
    return results

def remove_background_dynamic(image):
    """
    เลือกวิธีลบพื้นหลังที่ดีที่สุดแบบ dynamic
    - ถ้าเป็นภาพคน (ตรวจสอบเบื้องต้น) ใช้ MODNet ก่อน
    - ถ้าไม่สำเร็จ fallback backgroundremover > rembg
    - ถ้าไม่มีคืนภาพเดิม
    """
    # ลอง MODNet ก่อน
    img_modnet = remove_background_modnet_api(image)
    if img_modnet is not None:
        return img_modnet
    if BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove_bg_sota(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    if REMBG_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            img_no_bg = remove(img_rgba)
            return img_no_bg.convert("RGBA")
        except Exception:
            pass
    return image.convert("RGBA")

# ---------------- UI ----------------
st.set_page_config(page_title="AI Stylist", layout="centered")

st.markdown('<div class="main-title">👗 AI Stylist</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">อัปโหลดรูปภาพการแต่งตัวของคุณ ระบบจะวิเคราะห์สีเสื้อ/กางเกงและแนะนำสีที่เหมาะสม</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 เลือกรูปภาพการแต่งตัวของคุณ", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- ลบพื้นหลัง: ลอง rembg (bytes) > backgroundremover > rembg (bytes-in, bytes-out) > MODNet API > manual_remove_bg ---
    error_msgs = []
    image_nobg_bytes = None
    bg_method = ""
    # 1. rembg (bytes)
    try:
        image_nobg_bytes = remove_background_bytes_v2(image_bytes)
        if image_nobg_bytes is not None:
            bg_method = "rembg (bytes)"
    except Exception as e:
        error_msgs.append(f"rembg (bytes) error: {e}")
    # 2. backgroundremover
    if image_nobg_bytes is None and BGREMOVER_AVAILABLE:
        try:
            img_rgba = image.convert("RGBA")
            image_nobg_bytes = remove_bg_sota(img_rgba).convert("RGBA")
            if image_nobg_bytes is not None:
                bg_method = "backgroundremover (fallback)"
        except Exception as e:
            error_msgs.append(f"backgroundremover error: {e}")
    # 3. rembg (bytes-in, bytes-out)
    if image_nobg_bytes is None and REMBG_AVAILABLE:
        try:
            image_nobg_bytes = remove_background_bytes_v3(image_bytes)
            if image_nobg_bytes is not None:
                bg_method = "rembg (bytes-in, bytes-out) (fallback)"
        except Exception as e:
            error_msgs.append(f"rembg (bytes-in, bytes-out) error: {e}")
    # 4. MODNet API
    if image_nobg_bytes is None:
        try:
            image_nobg_bytes = remove_background_modnet_api(image)
            if image_nobg_bytes is not None:
                bg_method = "MODNet API (fallback)"
        except Exception as e:
            error_msgs.append(f"MODNet API error: {e}")
    # 5. Cloud API (fallback)
    if image_nobg_bytes is None:
        try:
            API_URL = "https://api.backgroundremover.io/v1/remove"
            API_TOKEN = "t42QYYRVh4PuHJCaUU5i5YWW"
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            files = {"image_file": ("image.png", buffered, "image/png")}
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            response = requests.post(API_URL, files=files, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if "image_base64" in result:
                    from base64 import b64decode
                    img_bytes = b64decode(result["image_base64"])
                    image_nobg_bytes = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                    bg_method = "Cloud API (backgroundremover.io)"
            else:
                error_msgs.append(f"Cloud API error: {response.status_code} {response.text}")
        except Exception as e:
            error_msgs.append(f"Cloud API error: {e}")
    # 6. manual_remove_bg (offline heuristic)
    if image_nobg_bytes is None:
        try:
            # ใช้ manual_remove_bg: threshold ครอบคลุมทุกโทน, tolerance 50
            image_nobg_bytes = manual_remove_bg(image, bg_color=(255,255,255), tolerance=50)
            if image_nobg_bytes is not None:
                bg_method = "manual_remove_bg (offline heuristic, all bg, tol=50, dilation)"
        except Exception as e:
            error_msgs.append(f"manual_remove_bg error: {e}")
    if image_nobg_bytes is None:
        bg_method = "rembg, backgroundremover, rembg(bytes-in), MODNet, Cloud API, manual_remove_bg fail"

    col1, col2 = st.columns([3,1])
    with col1:
        st.image(image, caption="ภาพต้นฉบับที่คุณอัปโหลด", use_container_width=True)
    with col2:
        if image_nobg_bytes is not None:
            st.image(checkerboard_bg(image_nobg_bytes.resize((120,120))), caption=f"Preview ตัดพื้นหลัง ({bg_method})", use_container_width=True)
        else:
            # If Cloud API DNS/network error, show a clear message
            cloud_api_dns_error = any(
                "NameResolutionError" in msg or "Failed to resolve" in msg or "Name or service not known" in msg
                for msg in error_msgs
            )
            if cloud_api_dns_error:
                st.error("ลบพื้นหลังไม่สำเร็จทุกวิธี\nCloud API: ไม่สามารถเชื่อมต่ออินเทอร์เน็ตหรือ DNS ได้\n\nโปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต หรือแก้ไข DNS ของ devcontainer/VM เช่น เพิ่ม nameserver 8.8.8.8 ใน /etc/resolv.conf แล้วลองใหม่อีกครั้ง\n\nรายละเอียด:\n" + "\n".join(error_msgs))
            else:
                st.error("ลบพื้นหลังไม่สำเร็จด้วยทุกวิธี\n" + "\n".join(error_msgs) if error_msgs else "rembg (bytes), backgroundremover, rembg (bytes-in), MODNet API ลบพื้นหลังไม่สำเร็จ")

    # --- Human Parsing: แยกเสื้อ/กางเกง ---
    mask = segment_clothes(image_nobg_bytes if image_nobg_bytes is not None else image)
    upper_img, lower_img = None, None
    if mask is not None:
        upper_img = extract_part(image_nobg_bytes if image_nobg_bytes is not None else image, mask, part_labels=[5])
        lower_img = extract_part(image_nobg_bytes if image_nobg_bytes is not None else image, mask, part_labels=[6])

    # --- วิเคราะห์สีและแนะนำสี (เสื้อ) ---
    st.markdown("<h4>👕 สีเสื้อ (Upper Clothes)</h4>", unsafe_allow_html=True)
    if upper_img is not None:
        upper_color = get_dominant_colors(upper_img, k=3)[0]
        upper_hex = rgb_to_hex(upper_color)
        upper_name = get_color_name_th(upper_color)
        st.markdown(f"<b>สีหลัก:</b> <span style='color:{upper_hex};font-weight:700;'>{upper_hex}</span> <b>{upper_name}</b>", unsafe_allow_html=True)
        # แนะนำสีที่เหมาะสม (complementary)
        import colorsys
        r, g, b = upper_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h2 = (h + 0.5) % 1.0
        comp_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(h2, s, v))
        comp_hex = rgb_to_hex(comp_rgb)
        comp_name = get_color_name_th(comp_rgb)
        st.markdown(f"<b>สีที่แนะนำ:</b> <span style='color:{comp_hex};font-weight:700;'>{comp_hex}</span> <b>{comp_name}</b>", unsafe_allow_html=True)
    else:
        st.info("ไม่พบส่วนเสื้อในภาพนี้")

    # --- วิเคราะห์สีและแนะนำสี (กางเกง) ---
    st.markdown("<h4>👖 สีกางเกง (Lower Clothes)</h4>", unsafe_allow_html=True)
    if lower_img is not None:
        lower_color = get_dominant_colors(lower_img, k=3)[0]
        lower_hex = rgb_to_hex(lower_color)
        lower_name = get_color_name_th(lower_color)
        st.markdown(f"<b>สีหลัก:</b> <span style='color:{lower_hex};font-weight:700;'>{lower_hex}</span> <b>{lower_name}</b>", unsafe_allow_html=True)
        # แนะนำสีที่เหมาะสม (complementary)
        import colorsys
        r, g, b = lower_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h2 = (h + 0.5) % 1.0
        comp_rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(h2, s, v))
        comp_hex = rgb_to_hex(comp_rgb)
        comp_name = get_color_name_th(comp_rgb)
        st.markdown(f"<b>สีที่แนะนำ:</b> <span style='color:{comp_hex};font-weight:700;'>{comp_hex}</span> <b>{comp_name}</b>", unsafe_allow_html=True)
    else:
        st.info("ไม่พบส่วนกางเกงในภาพนี้")

st.markdown('<div class="footer">👨‍💻 พัฒนาโดย นักศึกษาสาขาเทคโนโลยีสารสนเทศ</div>', unsafe_allow_html=True)
