import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import qrcode
import uuid
import pytesseract
import csv
import zipfile
import io
import threading
from datetime import datetime

# ==========================================
# ROI Configuration
# ==========================================
LOGO_CONFIG = {"h_end": 0.16, "w_start": 0.79}
LOGO_THRESHOLD = 5
TEXT_CONFIG = {"h_start": 0.1, "h_end": 0.16, "w_start": 0.79, "w_end": 1.0}
TEXT_KEYS = ["PRELIMINARY", "STAGE", "TEMPLATE", "CATEGORIES", "ALL"]
MIN_COLOR_PERCENTAGE = 50 # Fail if less than 50% of the artwork is colored

# ==========================================
# 1. Tesseract 引擎配置
# ==========================================
def init_tesseract():
    """自动探测并配置 Tesseract 路径"""
    paths = [
        r'/usr/bin/tesseract',            # Standard Debian/Ubuntu Linux path (Docker)
        r'/usr/local/bin/tesseract',      # Homebrew Mac path
        r'/opt/homebrew/bin/tesseract',   # Apple Silicon Mac path
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Tesseract-OCR', 'tesseract.exe')
    ]
    for p in paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    
    # Fallback to default PATH resolution if not explicitly found in the list
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

HAS_TESSERACT = init_tesseract()

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "assets": os.path.join(BASE_DIR, "01_Assets"),
    "inputs": os.path.join(BASE_DIR, "02_Inputs"),
    "outputs": os.path.join(BASE_DIR, "03_Output_Certs"),
    "diagnosis": os.path.join(BASE_DIR, "06_Diagnosis"),
    "database": os.path.join(BASE_DIR, "04_Database")
}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

# Global dict to store processing progress (in memory)
BATCH_PROGRESS = {}

def append_to_database(record):
    """追加记录到 Master Database CSV"""
    db_path = os.path.join(PATHS["database"], "database.csv")
    file_exists = os.path.isfile(db_path)
    
    try:
        with open(db_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Name", "Class", "School", "State", "Category", "Status", "Score", "CertFilename", "ArtworkURL"])
            writer.writerow(record)
    except Exception as e:
        print(f"Database Write Error: {e}")

def get_next_row_id():
    """获取下一个可用的数据库行号"""
    db_path = os.path.join(PATHS["database"], "database.csv")
    if not os.path.isfile(db_path):
        return 1
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f if line.strip())
        return count if count > 0 else 1
    except Exception:
        return 1

# ==========================================
# 2. 核心工具函数
# ==========================================
def get_roi(img, config):
    """根据比例裁剪感兴趣区域"""
    h, w = img.shape[:2]
    ys = int(h * config.get("h_start", 0))
    ye = int(h * config["h_end"])
    ws = int(w * config["w_start"])
    we = int(w * config.get("w_end", 1.0))
    return img[ys:ye, ws:we]

def match_logo(ref, target):
    """ORB 特征匹配逻辑"""
    if target.size == 0: return 0, None, [], [], []
    h_t = target.shape[0]
    ref_res = cv2.resize(ref, (int(ref.shape[1] * (h_t / ref.shape[0])), h_t))
    
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(ref_res, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    
    if des1 is None or des2 is None: return 0, ref_res, kp1, kp2, []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]
    return len(good), ref_res, kp1, kp2, good

def preprocess_for_ocr(img):
    """优化版：自适应亮度 + 边框填充"""
    if img is None or getattr(img, 'size', 0) == 0: return img
    
    resized = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    avg_brightness = np.mean(gray)
    if avg_brightness < 127:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    processed = cv2.medianBlur(binary, 3)
    pad = 40
    processed = cv2.copyMakeBorder(processed, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    return processed

def check_color_percentage(img):
    """
    检查画作的色彩比例。转换为 HSV 空间，计算饱和度 (Saturation) > 25 的像素比例。
    忽略右上角的 Logo 和文本区域，以防干扰。
    """
    if img is None or getattr(img, 'size', 0) == 0: return 0.0
    
    h, w = img.shape[:2]
    # 缩小图片以加快计算速度
    small_img = cv2.resize(img, (w // 4, h // 4))
    sh, sw = small_img.shape[:2]
    
    # 忽略右上角区域 (h_end = 0.16, w_start = 0.79)
    # 通过创建一个掩膜，将右上角的饱和度强制置为 0
    ignore_h_end = int(sh * 0.16)
    ignore_w_start = int(sw * 0.79)
    
    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    # S 通道表示色彩饱和度
    saturation = hsv[:, :, 1]
    
    # 将右上角置零
    saturation[0:ignore_h_end, ignore_w_start:sw] = 0
    
    # 计算有效绘画区域的总像素 (图像总像素 - 右上角像素)
    total_pixels = (sh * sw) - (ignore_h_end * (sw - ignore_w_start))
    
    # 大于 25 视为有颜色 (排除纯白、纯黑、铅笔灰)
    colored_pixels = np.sum(saturation > 25)
    
    if total_pixels <= 0: return 0.0
    percentage = (colored_pixels / total_pixels) * 100
    return percentage

# ==========================================
# 3. AI 审计引擎
# ==========================================
def run_ai_audit(image_path, name):
    try:
        img = cv2.imread(image_path)
        logo_ref = cv2.imread(os.path.join(PATHS["assets"], "logo_ref.png"))
        if img is None or logo_ref is None: return False, 0, "缺少图像文件"

        roi_l = get_roi(img, LOGO_CONFIG)
        score_l, ref_l, kp1, kp2, matches = match_logo(logo_ref, roi_l)

        score_t, ocr_res = 0, ""
        if HAS_TESSERACT:
            roi_t = get_roi(img, TEXT_CONFIG)
            processed_t = preprocess_for_ocr(roi_t)
            ocr_res = str(pytesseract.image_to_string(processed_t, config='--oem 3 --psm 6', lang='eng')).upper()
            score_t = sum(1 for k in TEXT_KEYS if k in ocr_res)
            cv2.imwrite(os.path.join(PATHS["diagnosis"], f"OCR_INPUT_{name}.jpg"), processed_t)
        else:
            return False, 0, "OCR 引擎未安装"

        is_pass = (score_l >= LOGO_THRESHOLD) and (score_t >= 1)
        
        # 色彩检测逻辑
        color_percentage = 0.0
        if is_pass:
            color_percentage = check_color_percentage(img)
            if color_percentage < MIN_COLOR_PERCENTAGE:
                is_pass = False
                color_fail = True
            else:
                color_fail = False
        else:
            color_fail = False

        status = "PASS" if is_pass else "FAIL"
        
        # 报告渲染
        viz = cv2.drawMatches(ref_l, kp1, roi_l, kp2, matches, None, flags=2)
        h_h = 240
        header = np.zeros((h_h, viz.shape[1], 3), dtype=np.uint8)
        ocr_preview = str(ocr_res)[:35].strip().replace('\n', ' ')
        info = [
            f"STUDENT: {name}", 
            f"LOGO SCORE: {score_l} (MIN:5)", 
            f"OCR READ: {ocr_preview}", 
            f"COLOR DENSITY: {color_percentage:.1f}% (MIN:{MIN_COLOR_PERCENTAGE}%)",
            f"AUDIT STATUS: {status}"
        ]
        
        for i, text in enumerate(info):
            color = (255, 255, 255)
            if "PASS" in text: color = (0, 255, 0)
            if "FAIL" in text: color = (0, 0, 255)
            cv2.putText(header, text, (20, 35 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        
        report = np.vstack((header, viz))
        cv2.imwrite(os.path.join(PATHS["diagnosis"], f"{status}_REPORT_{name}.jpg"), report)
        
        if is_pass:
            msg = "Success"
        elif color_fail:
            msg = f"Not enough color: {color_percentage:.1f}% (Minimum {MIN_COLOR_PERCENTAGE}%)"
        else:
            msg = "Logo or Text match failed (Ensure photo is clear and A4 proportioned)"
            
        return bool(is_pass), int(score_l) + int(score_t), str(msg)
    except Exception as e:
        return False, 0, str(e)

# ==========================================
# 4. 证书生成与路由
# ==========================================
def create_pdf(name, sub_id):
    try:
        img = Image.open(os.path.join(PATHS["assets"], "template.png")).convert("RGB")
        draw = ImageDraw.Draw(img)
        font_p = os.path.join(PATHS["assets"], "font.ttf")
        fs = 85
        font = ImageFont.truetype(font_p, fs)
        while draw.textbbox((0, 0), name, font=font)[2] > img.size[0] * 0.75:
            fs -= 5
            font = ImageFont.truetype(font_p, fs)
        w = draw.textbbox((0, 0), name, font=font)[2]
        draw.text(((img.size[0]-w)/2, 788), name, fill="black", font=font)
        qr = qrcode.make(f"https://cert.auth/v/{sub_id}").resize((180, 180))
        img.paste(qr, (img.size[0]-250, img.size[1]-250))
        
        safe_name = name.replace(' ', '_').replace('/', '_')
        fname = f"{sub_id}_{safe_name}.pdf"
        img.save(os.path.join(PATHS["outputs"], fname), "PDF")
        return fname
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

def create_pdf_batch(names, zip_path, zip_id):
    """Ultra-optimized RAM-based batch PDF generator to beat Render 100s timeout"""
    try:
        BATCH_PROGRESS[zip_id] = {"current": 0, "total": len(names), "status": "processing"}
        
        # 1. Pre-load Image and Font to memory ONCE
        base_img = Image.open(os.path.join(PATHS["assets"], "template.png")).convert("RGB")
        font_p = os.path.join(PATHS["assets"], "font.ttf")
        default_font = ImageFont.truetype(font_p, 85)
        max_width = base_img.size[0] * 0.75
        
        generated_files = []
        temp_zip = zip_path + ".tmp"
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, name in enumerate(names):
                if i % 10 == 0:
                    BATCH_PROGRESS[zip_id]["current"] = i
                # 2. Work entirely in RAM
                img = base_img.copy()
                draw = ImageDraw.Draw(img)
                
                # 3. Cache Font Logic (99% of names fit instantly in default font)
                w = draw.textbbox((0, 0), name, font=default_font)[2]
                if w <= max_width:
                    font = default_font
                else:
                    fs = 80
                    font = ImageFont.truetype(font_p, fs)
                    while draw.textbbox((0, 0), name, font=font)[2] > max_width and fs > 20:
                        fs -= 5
                        font = ImageFont.truetype(font_p, fs)
                    w = draw.textbbox((0, 0), name, font=font)[2]
                
                draw.text(((img.size[0]-w)/2, 788), name, fill="black", font=font)
                
                # 4. Fast QR logic
                u_id_hex = uuid.uuid4().hex
                sub_id = f"YAA-{u_id_hex[:5].upper()}"
                qr = qrcode.make(f"https://cert.auth/v/{sub_id}").resize((180, 180))
                img.paste(qr, (img.size[0]-250, img.size[1]-250))
                
                # 5. Sanitize Name
                safe_name = name.replace(' ', '_').replace('/', '_')
                fname = f"{sub_id}_{safe_name}.pdf"
                
                # 6. Save directly to RAM buffer, bypass SSD write perfectly!
                pdf_bytes = io.BytesIO()
                img.save(pdf_bytes, format="PDF")
                zipf.writestr(fname, pdf_bytes.getvalue())
                
                generated_files.append(fname)
        
        # 生成完毕后重命名，防止前端过早下载到不完整的 ZIP
        os.rename(temp_zip, zip_path)
        BATCH_PROGRESS[zip_id]["current"] = len(names)
        BATCH_PROGRESS[zip_id]["status"] = "done"
        return True, len(generated_files)
    except Exception as e:
        print(f"Batch PDF Error: {e}")
        BATCH_PROGRESS[zip_id]["status"] = "error"
        return False, 0

@app.route('/api/process', methods=['POST'])
def handle_request():
    """主接口：处理前端发送的姓名(或CSV)、频道和作品照片"""
    channel = request.form.get('channel', 'Individual')
    file = request.files.get('artwork')
    
    # ------------------
    # 批量上传处理通道 (School + CSV)
    # ------------------
    if channel == "School" and file and file.filename.endswith('.csv'):
        names = []
        try:
            # 读取 CSV 内容
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.reader(stream)
            
            for row in csv_input:
                # 学校模板有很多行说明文字。真实数据行的特点是：第一列(BIL)是纯数字，第二列(NAMA)是名字。
                if len(row) > 1:
                    bil = row[0].strip()
                    if bil.isdigit():
                        student_name = row[1].strip()
                        # 跳过模板里的例子 "NAMA MURID ANDA DALAM UPPERCASE"
                        if student_name and "NAMA MURID ANDA" not in student_name:
                            names.append(student_name)
        except Exception as e:
            return jsonify({"status": "fail", "message": f"CSV读取失败: {str(e)}"})

        if not names:
            return jsonify({"status": "fail", "message": "CSV文件为空或格式不正确"})

        # 生成所有证书并打包为 ZIP (使用后台线程)
        zip_filename = f"Batch_Certs_{uuid.uuid4().hex[:6]}.zip"
        zip_path = os.path.join(PATHS["outputs"], zip_filename)
        
        # 启动后台线程执行批量生成, 避免前端 100秒超时断连!
        thread = threading.Thread(target=create_pdf_batch, args=(names, zip_path, zip_filename), daemon=True)
        thread.start()
        
        return jsonify({
            "status": "success",
            "score": 100,
            "file": zip_filename, # 返回 ZIP 文件名供前端调用 progress 接口
            "message": f"正在后台为您生成 {len(names)} 份证书"
        })

    # ------------------
    # 单个上传处理通道 (Individual or Single School Pass)
    # ------------------
    name = request.form.get('name', 'UNKNOWN').upper()
    
    # New individual data fields
    class_name = request.form.get('class_name', '')
    category = request.form.get('category', '')
    school = request.form.get('school', '')
    state = request.form.get('state', '')
    
    u_id_hex = uuid.uuid4().hex
    t_path = os.path.join(PATHS["inputs"], f"up_{u_id_hex[:6]}.jpg")
    if file: 
        file.save(t_path)
    
    passed, score, msg = False, 0, "等待处理"
    
    if channel == "Individual":
        passed, score, msg = run_ai_audit(t_path, name)
    else:
        passed, score, msg = True, 100, "校方通道自动通过"
    
    pdf_filename = None
    if passed:
        u_id_hex = uuid.uuid4().hex
        sub_id = f"YAA-{u_id_hex[:5].upper()}"
        pdf_filename = create_pdf(name, sub_id)
        
    # Log to Database if Individual
    if channel == "Individual":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_str = "PASS" if passed else "FAIL"
        
        # Generate the new filename with row ID and safe name
        row_id = get_next_row_id()
        safe_name_for_file = name.replace(' ', '_').replace('/', '_')
        new_filename = f"{row_id:03d}_{safe_name_for_file}.jpg"
        new_path = os.path.join(PATHS["inputs"], new_filename)
        
        # Rename the original uploaded file to the new permanent name
        if os.path.exists(t_path):
            os.rename(t_path, new_path)
        
        # Generate the dynamic public URL for the newly renamed artwork
        artwork_url = f"{request.host_url}api/artwork/{new_filename}"
        
        record = [timestamp, name, class_name, school, state, category, status_str, score, pdf_filename or "N/A", artwork_url]
        append_to_database(record)
    
    return jsonify({
        "status": "success" if passed else "fail",
        "score": score,
        "file": pdf_filename,
        "message": msg
    })

@app.route('/api/progress/<zip_id>', methods=['GET'])
def get_progress(zip_id):
    """获取后台批量生成的进度"""
    if zip_id in BATCH_PROGRESS:
        return jsonify(BATCH_PROGRESS[zip_id])
    return jsonify({"status": "unknown"}), 404

@app.route('/api/download/<filename>', methods=['GET'])
def download_cert(filename):
    """供前端下载生成的证书 PDF 或 ZIP"""
    if not os.path.exists(os.path.join(PATHS["outputs"], filename)):
        return "<h1>文件尚未生成完成</h1><p>请等待进度条达到 100% 后重试。</p>", 404
        
    return send_from_directory(PATHS["outputs"], filename, as_attachment=True)

@app.route('/api/artwork/<filename>', methods=['GET'])
def get_artwork(filename):
    """Serve the original uploaded cropped artwork for viewing via CSV link"""
    if not os.path.exists(os.path.join(PATHS["inputs"], filename)):
        return "<h1>Artwork Not Found</h1><p>The requested image does not exist.</p>", 404
        
    return send_from_directory(PATHS["inputs"], filename)

@app.route('/api/admin/database', methods=['GET'])
def download_database():
    """Admin Dashboard: Securely download the master database CSV"""
    # Require a basic token in the Authorization header
    auth_header = request.headers.get('Authorization')
    
    # Hardcoded admin password for simplicity (can be moved to .env later)
    if not auth_header or auth_header != "Bearer yaa_admin_2026":
        return jsonify({"status": "fail", "message": "Unauthorized: Invalid admin password"}), 401
        
    db_file = "database.csv"
    db_dir = PATHS["database"]
    
    if not os.path.exists(os.path.join(db_dir, db_file)):
        return jsonify({"status": "fail", "message": "Database file not found yet"}), 404
        
    return send_from_directory(db_dir, db_file, as_attachment=True)

if __name__ == '__main__':
    print("\n🚀 YAA 审计服务器已启动 | 对接门户模式 (Fixed)")
    print(f"📂 诊断报告保存至: {PATHS['diagnosis']}")
    app.run(host='0.0.0.0', port=5001)