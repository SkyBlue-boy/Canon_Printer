import streamlit as st
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import shutil
import time
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
from Levenshtein import distance as lev # 필수 라이브러리

# ==========================================
# [설정] 경로 및 환경 설정
# ==========================================
# 모듈 경로 추가
sys.path.append(os.path.join(os.getcwd(), 'anomaly', 'PatchGuard'))

# PatchGuard 관련 Import
from patchguard import PatchGuard

# GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과 저장 루트 폴더
RESULT_ROOT_DIR = "Results"

# ==========================================
# [설정] 모델 가중치 경로
# ==========================================
WEIGHTS = {
    "pg_printer": "patchguard_mvtec_printer.pth",
    "pg_lcd": "patchguard_mvtec_lcd.pth",
    "yolo_icon": "best.pt",
    "sr_model": "EDSR_x4.pb"
}

# ==========================================
# [설정] 크롭 좌표 및 임계값
# ==========================================
CROPS = {
    "printer": {"x1": 300, "y1": 200, "x2": 1280, "y2": 710},
    "lcd":     {"x1": 730, "y1": 240, "x2": 1280, "y2": 620},
    "ocr":     {"x1": 310, "y1": 220, "x2": 780,  "y2": 660} # Source와 동일 (310, 220, 780, 660)
}

THRESHOLDS = {
    "printer_score": 0.8,    # 이상이면 FAIL
    "lcd_score": -1.0,       # 이상이면 FAIL # -3.0, -1.0
    "icon_dist_px": 5.0      # 이상이면 FAIL
}

VIS_PARAMS = {
    "printer": {"vmin": 0, "vmax": 0.8, "sigma": 8, "colormap": cv2.COLORMAP_JET},
    "lcd":     {"vmin": -3.3, "vmax": -3.0, "sigma": 8, "colormap": cv2.COLORMAP_HOT} 
}

# ==========================================
# [설정] 기종 분류용 키워드 사전 (원본 100% 반영)
# ==========================================

# [절대 규칙용] 이 단어가 보이면 점수 계산 없이 즉시 해당 언어로 확정
PRIORITY_KEYWORDS = {
    "Japan": [
        'ホーム', '戻る', 'コピー', 'メニュー', 
        '木一么', '木一山', '一么', '本一么', '木一', '未一么', # 홈(Home) 오인식
        '戻う', '戻話', '戻コ' # 돌아가기(Back) 오인식
    ],
    "Taiwan": [
        '首頁', '上一頁', '身份證', 
        '首真', '首貞', '首道', '首夏', '首項', # 홈(Home) 오인식
        '状熊確', '状慈確', '状魅' # 대만어 특유의 깨짐 패턴
    ],
    "China": [
        'ID卡复印', '主页', '复印', '身份证', '返回' # 간체는 형태가 뚜렷함
    ],
    "Korea": [
        '상황확인', '이전화면', '신분증', '복사' # 한글은 겹칠 일이 없음
    ]
}

# [점수 계산용] 일반 키워드
KR_WORDS = ['홈', '상황확인', '이전화면', '복사', '스캔', '인쇄']

JP_WORDS = [
    'ホーム', '戻る', '状況確認', 'コピー', 
    '木一么', '木一山', '一么', '本一么', '木一', '木一△', '未一么',
    '状况確', '状说確', '状耀', '确器', '状况确忍', '状况罐器', '状况確器', '状况確記',
    '兄確露', '状况確热', '状况碾录', '状沉確热', '状確图', '状倪麗器', '状况確露'
]

TW_WORDS = [
    '首頁', '狀態確認', '上一步', 
    '首真', '首貞', '首道', '状熊確', '状慈確', '状态确', '状魅'
]

CN_WORDS = ['ID卡复印', '主页', '状态确认', '上一步', '复印']
ENG_WORDS = ['Home', 'ID', 'Card', 'Copy', 'Back', 'Status', 'Monitor']

# YOLO 클래스 ID
CLASS_BACK_ICON = 1
CLASS_ID_ICON = 7

# ==========================================
# [Class] PatchGuard Wrapper
# ==========================================
class PatchGuardWrapper:
    def __init__(self, weight_path, class_type):
        self.device = DEVICE
        self.class_type = class_type
        
        self.args = {
            "hf_path": 'vit_small_patch14_dinov2.lvd142m',
            "feature_layers": [12],
            "reg_layers": [6, 9, 12],
            "image_size": 256,
            "hidden_dim": 2048,
            "dsc_layers": 1,
            "dsc_heads": 4
        }
        
        class Args:
            def __init__(self, d):
                for k, v in d.items(): setattr(self, k, v)
        
        self.model = PatchGuard(Args(self.args), self.device).to(self.device)
        self.model.eval()
        
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            print(f"[PatchGuard] Loaded {class_type} weights from {weight_path}")
        else:
            st.error(f"Weights not found: {weight_path}")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

    def infer(self, img_bgr):
        c = CROPS[self.class_type]
        crop = img_bgr[c['y1']:c['y2'], c['x1']:c['x2']]
        if crop.size == 0: return True, 999.0, img_bgr

        img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inp = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.model(inp)
        
        batch_size, num_patches = scores.shape
        patches_per_side = int(np.sqrt(num_patches))
        patch_scores = scores[0].reshape((patches_per_side, patches_per_side))
        
        h, w = crop.shape[:2]
        scores_interpolated = torch.nn.functional.interpolate(
            patch_scores.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        localization = gaussian_filter(scores_interpolated, sigma=VIS_PARAMS[self.class_type]['sigma'])
        max_score = localization.max()
        
        threshold = THRESHOLDS[f"{self.class_type}_score"]
        is_fail = max_score >= threshold

        vis_img = None
        if is_fail:
            params = VIS_PARAMS[self.class_type]
            vmin, vmax = params['vmin'], params['vmax']

            norm_map = (localization - vmin) / (vmax - vmin)
            norm_map = np.clip(norm_map, 0, 1)
            norm_map = np.uint8(255 * norm_map)

            heatmap = cv2.applyColorMap(norm_map, params['colormap'])
            vis_crop = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)
            
            vis_img = img_bgr.copy()
            vis_img[c['y1']:c['y2'], c['x1']:c['x2']] = vis_crop

            # Text overlay
            text_color = (0, 0, 255) 
            cv2.putText(vis_img, f"{self.class_type.upper()} FAIL: {max_score:.2f}", 
                        (c['x1'], c['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        return is_fail, max_score, vis_img

# ==========================================
# [Helpers] Model Loaders
# ==========================================
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_ocr_model():
    return {
        'ch': PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=False, show_log=False),
        'korean': PaddleOCR(use_angle_cls=False, lang='korean', use_gpu=False, show_log=False),
        'japan': PaddleOCR(use_angle_cls=False, lang='japan', use_gpu=False, show_log=False)
    }

@st.cache_resource
def load_sr_model():
    """Super Resolution 모델 로드 (SR_MODEL_PATH = WEIGHTS['sr_model'])"""
    sr_path = WEIGHTS['sr_model']
    if not os.path.exists(sr_path):
        print(f"[경고] SR 모델 파일이 없습니다: {sr_path}")
        return None

    try:
        from cv2 import dnn_superres
        sr_impl = dnn_superres.DnnSuperResImpl_create()
        sr_impl.readModel(sr_path)
        sr_impl.setModel("edsr", 4)
        
        # GPU 설정 시도
        try:
            sr_impl.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr_impl.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print(">>> SR Model is set to use GPU (CUDA).")
        except Exception as e:
            print(f"[Warning] OpenCV CUDA 설정 실패 (CPU로 동작합니다): {e}")
        
        return sr_impl
    except Exception as e:
        print(f"[Error] SR 모델 로드 실패: {e}")
        return None

# ==========================================
# [Logic] Icon Position Check
# ==========================================
def check_icon_position(img_bgr, model, conf_thresh):
    results = model(img_bgr, verbose=False, conf=conf_thresh)
    result = results[0]
    
    check_pairs = [("Home_button", "Home_icon"), ("Status_button", "Status_icon"),
                   ("Back_button", "Back_icon"), ("id_button", "id_icon")]
    
    detected_objects = {}
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy()
        if cls_name not in detected_objects: detected_objects[cls_name] = []
        detected_objects[cls_name].append(xyxy)
        
    is_fail = False
    vis_img = img_bgr.copy()
    max_dist = 0.0
    
    def get_center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    for btn, icon in check_pairs:
        if btn in detected_objects and icon in detected_objects:
            btn_box = detected_objects[btn][0]
            icon_box = detected_objects[icon][0]
            c1, c2 = get_center(btn_box), get_center(icon_box)
            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist > max_dist: max_dist = dist
            
            color = (0, 255, 0)
            if dist > THRESHOLDS['icon_dist_px']:
                is_fail = True
                color = (0, 0, 255)
            
            cv2.circle(vis_img, (int(c1[0]), int(c1[1])), 3, (255,0,0), -1)
            cv2.circle(vis_img, (int(c2[0]), int(c2[1])), 3, (0,0,255), -1)
            cv2.line(vis_img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 2)
    
    return is_fail, max_dist, vis_img if is_fail else None

# ==========================================
# [Logic] Printer Classification (Full Implementation)
# ==========================================
def preprocess_for_ocr(crop_img, sr_model):
    if crop_img is None or crop_img.size == 0: return crop_img
    
    # 1. SR Model이 있으면 사용
    if sr_model is not None:
        try:
            upscaled = sr_model.upsample(crop_img)
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            return gray
        except Exception:
            pass

    # 2. Fallback: 기존 Resize + Sharpen
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(scaled, -1, kernel)
    return sharpened

def check_priority_match(ocr_texts, priority_dict):
    clean_texts = [t.replace(" ", "") for t in ocr_texts if t.strip()]
    
    for k in priority_dict["Japan"]:
        for txt in clean_texts:
            if k in txt: return "Japan"
            
    for k in priority_dict["Taiwan"]:
        for txt in clean_texts:
            if k in txt: return "Taiwan"
            
    for k in priority_dict["China"]:
        for txt in clean_texts:
            if k in txt: return "China"
            
    for k in priority_dict["Korea"]:
        for txt in clean_texts:
            if k in txt: return "Korea"
            
    return None

def calculate_match_score(ocr_texts, keywords):
    total_score = 0.0
    matched_words = []
    clean_texts = [t.replace(" ", "") for t in ocr_texts if t.strip()]

    for k in keywords:
        clean_k = k.replace(" ", "")
        for txt in clean_texts:
            if clean_k in txt:
                total_score += 1.0
                matched_words.append(k)
                break
            if len(clean_k) >= 3:
                d = lev(txt, clean_k)
                L = max(len(txt), len(clean_k))
                if (1.0 - d/L) > 0.65:
                    total_score += 0.8
                    matched_words.append(k)
                    break
    return total_score, matched_words

def determine_language_logic(readers, crop_img):
    """
    원본 코드의 determine_language_logic 완전 구현
    """
    extracted_texts = {
        'Korea': [],
        'Japan': [],
        'China': [], 
    }
    all_texts_combined = []

    # 한국어 모델
    try:
        res = readers['korean'].ocr(crop_img, cls=False)
        if res and res[0]: 
            txts = [line[1][0] for line in res[0]]
            extracted_texts['Korea'] = txts
            all_texts_combined.extend(txts)
    except: pass

    # 일본어 모델
    try:
        res = readers['japan'].ocr(crop_img, cls=False)
        if res and res[0]: 
            txts = [line[1][0] for line in res[0]]
            extracted_texts['Japan'] = txts
            all_texts_combined.extend(txts)
    except: pass

    # 중국어(통합) 모델
    try:
        res = readers['ch'].ocr(crop_img, cls=False)
        if res and res[0]: 
            txts = [line[1][0] for line in res[0]]
            extracted_texts['China'] = txts
            all_texts_combined.extend(txts)
    except: pass

    # 1. 절대 규칙
    priority_lang = check_priority_match(all_texts_combined, PRIORITY_KEYWORDS)
    if priority_lang:
        final_txt = extracted_texts.get(priority_lang, extracted_texts['China'])
        if not final_txt: final_txt = extracted_texts['China']
        return priority_lang, final_txt

    # 2. 점수 경쟁
    scores = {}
    score_kr, _ = calculate_match_score(extracted_texts['Korea'], KR_WORDS)
    score_jp, _ = calculate_match_score(extracted_texts['Japan'], JP_WORDS)
    
    ch_texts = extracted_texts['China']
    score_cn, matches_cn = calculate_match_score(ch_texts, CN_WORDS)
    score_tw, matches_tw = calculate_match_score(ch_texts, TW_WORDS)
    score_en, matches_en = calculate_match_score(ch_texts, ENG_WORDS)

    scores['Korea'] = score_kr
    scores['Japan'] = score_jp
    scores['China'] = score_cn
    scores['Taiwan'] = score_tw
    scores['English'] = score_en

    # 보정: 대만 vs 중국
    if score_tw > 0 and score_tw == score_cn:
        if any(w in ['狀態確認', '首頁'] for w in matches_tw): 
            scores['China'] -= 0.1
        elif any(w in ['主页', 'ID卡复印'] for w in matches_cn):
            scores['Taiwan'] -= 0.1

    # 보정: 영어 vs 중국어
    if score_cn > 0 and score_en > 0:
        if score_cn >= 1.0:
            scores['English'] = 0

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]

    if best_lang == 'Korea':
        final_texts = extracted_texts['Korea']
    elif best_lang == 'Japan':
        final_texts = extracted_texts['Japan']
    else:
        final_texts = extracted_texts['China']

    if not final_texts: 
        final_texts = extracted_texts['China']

    if best_score < 0.8:
        return "No_label", final_texts

    return best_lang, final_texts

def determine_function_with_yolo(model, img, conf_thresh):
    """
    원본의 determine_function_with_yolo와 동일, conf_thresh만 연동
    """
    results = model(img, verbose=False, conf=conf_thresh)
    found_classes = []
    for result in results:
        for box in result.boxes:
            found_classes.append(int(box.cls[0]))
    
    if CLASS_ID_ICON in found_classes: return "id"
    elif CLASS_BACK_ICON in found_classes: return "back"
    return "unknown"

def classify_printer_model(img_bgr, readers, yolo_model, sr_model, conf_thresh):
    """
    process_directory 내부 로직을 단일 이미지 처리에 맞게 래핑
    """
    # 1. Fixed Crop
    c = CROPS['ocr']
    crop = img_bgr[c['y1']:c['y2'], c['x1']:c['x2']]
    
    # 2. Preprocess (SR or Basic)
    processed_crop = preprocess_for_ocr(crop, sr_model)

    # 3. Determine Language
    lang_result, ocr_texts = determine_language_logic(readers, processed_crop)
    
    final_prediction = "No_label"

    if lang_result == "No_label":
        final_prediction = "No_label" # UI 표시용 변경 # Unknown
    elif lang_result in ["Korea", "Japan", "Taiwan"]:
        final_prediction = lang_result
    elif lang_result in ["China", "English"]:
        # 4. Check Function (YOLO -> OCR fallback)
        func = determine_function_with_yolo(yolo_model, img_bgr, conf_thresh)
        if func == "unknown":
            score_id, _ = calculate_match_score(ocr_texts, ['ID', 'Copy', 'Card', '复印', 'ID卡'])
            if score_id >= 0.8: func = "id"
            else: func = "back"
        final_prediction = f"{lang_result}_{func}"
        
    return final_prediction

# ==========================================
# [Streamlit] Main App
# ==========================================
def main():
    st.set_page_config(page_title="Printer QA System", layout="wide")
    st.title("🖨️ Canon Printer QA Inspection System")
    st.markdown("---")

    with st.spinner("Loading AI Models..."):
        try:
            pg_printer = PatchGuardWrapper(WEIGHTS["pg_printer"], "printer")
            pg_lcd = PatchGuardWrapper(WEIGHTS["pg_lcd"], "lcd")
            yolo_icon = load_yolo_model(WEIGHTS["yolo_icon"])
            ocr_readers = load_ocr_model()
            sr_model = load_sr_model() # SR 모델 로드
            st.success("✅ All Models Loaded Successfully!")
        except Exception as e:
            st.error(f"❌ Model Loading Failed: {e}")
            return

    st.sidebar.header("Settings")
    conf_thresh = st.sidebar.slider("YOLO Confidence", 0.0, 1.0, 0.4)
    
    uploaded_files = st.file_uploader("Upload Test Images", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
    
    if st.button("Start Inspection") and uploaded_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULT_ROOT_DIR, f"Run_{timestamp}")
        fail_dir = os.path.join(save_dir, "FAIL")
        pass_dir = os.path.join(save_dir, "PASS")
        
        os.makedirs(fail_dir, exist_ok=True)
        os.makedirs(pass_dir, exist_ok=True)
        
        log_data = []
        pass_gallery_data = {} 

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            filename = uploaded_file.name
            file_root, file_ext = os.path.splitext(filename)
            
            status_text.text(f"Processing {filename}...")
            
            defect_reasons = []
            final_vis_img = img_bgr.copy()
            
            # 1. PatchGuard (Printer)
            is_fail_ptr, score_ptr, vis_ptr = pg_printer.infer(img_bgr)
            if is_fail_ptr:
                defect_reasons.append("Intrusion")
                final_vis_img = vis_ptr 

            # 2. PatchGuard (LCD)
            is_fail_lcd, score_lcd, vis_lcd = pg_lcd.infer(img_bgr)
            if is_fail_lcd:
                defect_reasons.append("Reflection")
                c = CROPS['lcd']
                final_vis_img[c['y1']:c['y2'], c['x1']:c['x2']] = vis_lcd[c['y1']:c['y2'], c['x1']:c['x2']]
                cv2.putText(final_vis_img, f"LCD FAIL: {score_lcd:.2f}", 
                            (c['x1'], c['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 3. Icon Position
            is_fail_pos, max_dist, vis_pos = check_icon_position(img_bgr, yolo_icon, conf_thresh)
            if is_fail_pos:
                defect_reasons.append("Position")
                if vis_pos is not None: final_vis_img = vis_pos
            
            # Result Processing
            detected_model = "-"
            result_status = "PASS"
            
            if defect_reasons:
                result_status = "FAIL"
                tags = "_".join(defect_reasons)
                new_filename = f"{file_root}_FAIL_[{tags}]{file_ext}"
                vis_filename = f"{file_root}_FAIL_[{tags}]_vis{file_ext}"
                
                cv2.imwrite(os.path.join(fail_dir, new_filename), img_bgr)
                cv2.imwrite(os.path.join(fail_dir, vis_filename), final_vis_img)
                
            else:
                # 4. Classification (Full Logic)
                detected_model = classify_printer_model(img_bgr, ocr_readers, yolo_icon, sr_model, conf_thresh)
                
                model_dir = os.path.join(pass_dir, detected_model)
                os.makedirs(model_dir, exist_ok=True)
                save_path = os.path.join(model_dir, filename)
                cv2.imwrite(save_path, img_bgr)
                
                if detected_model not in pass_gallery_data:
                    pass_gallery_data[detected_model] = []
                pass_gallery_data[detected_model].append(save_path)
            
            log_data.append({
                "Image Name": filename,
                "Result": result_status,
                "Defects": ", ".join(defect_reasons) if defect_reasons else "-",
                "Detected Model": detected_model,
                "Scores": f"P:{score_ptr:.2f}, L:{score_lcd:.2f}, D:{max_dist:.1f}"
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # CSV 저장
        df = pd.DataFrame(log_data)
        csv_path = os.path.join(save_dir, "inspection_log.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        status_text.success(f"Inspection Complete! Saved to {save_dir}")
        
        # --- UI Result Display ---
        st.header("📊 Inspection Results")
        
        total = len(df)
        fails = len(df[df['Result'] == 'FAIL'])
        passes = total - fails
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", total)
        col2.metric("Passed", passes, delta_color="normal")
        col3.metric("Failed", fails, delta_color="inverse")
        
        st.subheader("📋 Inspection Log")
        st.dataframe(df)
        
        # Failure Gallery
        st.subheader("❌ Failure Gallery")
        fail_images = [f for f in os.listdir(fail_dir) if "_vis" in f]
        if fail_images:
            cols = st.columns(3)
            for i, img_name in enumerate(fail_images):
                img_path = os.path.join(fail_dir, img_name)
                with cols[i % 3]:
                    st.image(img_path, caption=img_name, width="stretch")
        else:
            st.info("No failures detected. Perfect!")

        # Pass Gallery
        st.subheader("✅ Pass Gallery (Sorted by Model)")
        if pass_gallery_data:
            model_names = sorted(pass_gallery_data.keys())
            tabs = st.tabs(model_names)
            
            for tab, model_name in zip(tabs, model_names):
                with tab:
                    img_paths = pass_gallery_data[model_name]
                    st.write(f"**Count:** {len(img_paths)} images")
                    
                    cols = st.columns(3)
                    for i, p_path in enumerate(img_paths):
                        display_img = cv2.cvtColor(cv2.imread(p_path), cv2.COLOR_BGR2RGB)
                        with cols[i % 3]:
                            st.image(display_img, caption=os.path.basename(p_path), width="stretch")
        else:
            st.warning("No images passed the inspection.")

if __name__ == "__main__":
    main()