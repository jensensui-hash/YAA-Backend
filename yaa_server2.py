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

# ==========================================
# ROI Configuration
# ==========================================
LOGO_CONFIG = {"h_end": 0.16, "w_start": 0.79}
LOGO_THRESHOLD = 5
TEXT_CONFIG = {"h_start": 0.1, "h_end": 0.16, "w_start": 0.79, "w_end": 1.0}
TEXT_KEYS = ["PRELIMINARY", "STAGE", "TEMPLATE", "CATEGORIES", "ALL"]

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
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "assets": os.path.join(BASE_DIR, "01_Assets"),
    "inputs": os.path.join(BASE_DIR, "02_Inputs"),
    "outputs": os.path.join(BASE_DIR, "03_Output_Certs"),
    "diagnosis": os.path.join(BASE_DIR, "06_Diagnosis")
}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

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
        status = "PASS" if is_pass else "FAIL"
        
        # 报告渲染
        viz = cv2.drawMatches(ref_l, kp1, roi_l, kp2, matches, None, flags=2)
        h_h = 240
        header = np.zeros((h_h, viz.shape[1], 3), dtype=np.uint8)
        ocr_preview = str(ocr_res)[:35].strip().replace('\n', ' ')
        info = [f"STUDENT: {name}", f"LOGO SCORE: {score_l} (MIN:5)", f"OCR READ: {ocr_preview}", f"AUDIT STATUS: {status}"]
        for i, text in enumerate(info):
            color = (255, 255, 255)
            if "PASS" in text: color = (0, 255, 0)
            if "FAIL" in text: color = (0, 0, 255)
            cv2.putText(header, text, (20, 50 + i*45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        
        report = np.vstack((header, viz))
        cv2.imwrite(os.path.join(PATHS["diagnosis"], f"{status}_REPORT_{name}.jpg"), report)
        
        msg = "审计通过" if is_pass else "审计未通过：校徽或文字匹配失败"
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
        fname = f"{sub_id}_{name.replace(' ', '_')}.pdf"
        img.save(os.path.join(PATHS["outputs"], fname), "PDF")
        return fname
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

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

        # 生成所有证书并打包为 ZIP
        zip_filename = f"Batch_Certs_{uuid.uuid4().hex[:6]}.zip"
        zip_path = os.path.join(PATHS["outputs"], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for n in names:
                u_id_hex = uuid.uuid4().hex
                sub_id = f"YAA-{u_id_hex[:5].upper()}"
                pdf_name = create_pdf(n, sub_id)
                if pdf_name:
                    pdf_full_path = os.path.join(PATHS["outputs"], pdf_name)
                    zipf.write(pdf_full_path, arcname=pdf_name)
                    
        return jsonify({
            "status": "success",
            "score": 100,
            "file": zip_filename, # 返回 ZIP 文件名供前端下载
            "message": f"成功批量生成 {len(names)} 份证书"
        })

    # ------------------
    # 单个上传处理通道 (Individual or Single School Pass)
    # ------------------
    name = request.form.get('name', 'UNKNOWN').upper()
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
    
    return jsonify({
        "status": "success" if passed else "fail",
        "score": score,
        "file": pdf_filename,
        "message": msg
    })

@app.route('/api/download/<filename>', methods=['GET'])
def download_cert(filename):
    """供前端下载生成的证书 PDF"""
    return send_from_directory(PATHS["outputs"], filename, as_attachment=True)

if __name__ == '__main__':
    print("\n🚀 YAA 审计服务器已启动 | 对接门户模式 (Fixed)")
    print(f"📂 诊断报告保存至: {PATHS['diagnosis']}")
    app.run(host='0.0.0.0', port=5001)