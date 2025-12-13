import os

# =========================================================
# [설정] GPU 1번 사용
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
# dnn_superres는 아래 try-except 블록에서 안전하게 임포트합니다.
import csv
import re
import logging
import numpy as np
from Levenshtein import distance as lev
from paddleocr import PaddleOCR
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import paddle

# ---------------------------------------------------------
# [설정] PaddleOCR은 CPU로 고정 (GPU 메모리 충돌 방지)
# ---------------------------------------------------------
paddle.set_device('cpu')
print(">>> PaddlePaddle is set to use CPU.")

# ==========================================
# [설정] 0. Super Resolution 모델 (GPU 설정)
# ==========================================
SR_MODEL_PATH = r"/home/shlee/Final_code/Canon_Printer/EDSR_x4.pb"
sr = None

try:
    from cv2 import dnn_superres
    sr_impl = dnn_superres.DnnSuperResImpl_create()
    
    if os.path.exists(SR_MODEL_PATH):
        print(f">>> Loading Super Resolution Model: {SR_MODEL_PATH}")
        sr_impl.readModel(SR_MODEL_PATH)
        sr_impl.setModel("edsr", 4)
        
        # [중요] SR 모델은 GPU(CUDA)로 실행하여 속도 확보
        try:
            sr_impl.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr_impl.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print(">>> SR Model is set to use GPU (CUDA).")
        except Exception as e:
            print(f"[Warning] OpenCV CUDA 설정 실패 (CPU로 동작합니다): {e}")
        
        sr = sr_impl
    else:
        print(f"[경고] SR 모델 파일이 없습니다: {SR_MODEL_PATH}")
        sr = None
except Exception as e:
    print(f"[Error] SR 모델 로드 실패: {e}")
    sr = None

# ==========================================
# [설정] 1. 고정 크롭 좌표
# ==========================================
FIXED_X_START, FIXED_Y_START = 310, 220
FIXED_X_END,   FIXED_Y_END   = 780, 660

# ==========================================
# [설정] 2. 키워드 사전 (콤마 오류 수정 및 그룹핑)
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

# ==========================================
# [설정] 3. YOLO 클래스 ID
# ==========================================
CLASS_BACK_ICON = 1
CLASS_ID_ICON = 7

# ==========================================
# [함수] 전처리
# ==========================================
def preprocess_for_ocr(crop_img):
    if crop_img is None or crop_img.size == 0: return crop_img
    
    if sr is not None:
        try:
            upscaled = sr.upsample(crop_img)
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            return gray
        except Exception:
            pass

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(scaled, -1, kernel)
    return sharpened

# ==========================================
# [함수] 키워드 매칭 여부 (절대 규칙용)
# ==========================================
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

# ==========================================
# [함수] 점수 계산 (일반 경쟁용)
# ==========================================
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

# ==========================================
# [핵심] 언어 판별 로직 (텍스트 분리 저장 기능 추가)
# ==========================================
def determine_language_logic(readers, crop_img):
    """
    1. 각 모델별 결과를 따로 저장
    2. 절대 규칙(Priority) 확인
    3. 점수 경쟁(Competition)
    4. 승리한 언어의 텍스트만 반환
    """
    # ---------------------------------------------
    # 1. 모델별 결과 분리 저장
    # ---------------------------------------------
    extracted_texts = {
        'Korea': [],
        'Japan': [],
        'China': [], # China, Taiwan, English, No_label은 이 결과를 공유
    }
    
    # 모든 텍스트 합침 (절대 규칙 판단용)
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

    # ---------------------------------------------
    # 2. [절대 규칙] 결정적인 단어가 있으면 즉시 리턴
    # ---------------------------------------------
    priority_lang = check_priority_match(all_texts_combined, PRIORITY_KEYWORDS)
    if priority_lang:
        # 우선순위로 결정된 경우, 해당 언어 모델의 텍스트를 반환
        final_txt = extracted_texts.get(priority_lang, extracted_texts['China'])
        if not final_txt: final_txt = extracted_texts['China'] # fallback
        return priority_lang, final_txt

    # ---------------------------------------------
    # 3. [점수 경쟁]
    # ---------------------------------------------
    scores = {}
    
    # 각 언어 점수는 '자기네 모델'이 읽은 텍스트로 채점
    score_kr, _ = calculate_match_score(extracted_texts['Korea'], KR_WORDS)
    score_jp, _ = calculate_match_score(extracted_texts['Japan'], JP_WORDS)
    
    # 중/대/영은 ch 모델 결과 사용
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

    # [핵심] 승리한 언어의 텍스트만 선택하여 저장
    if best_lang == 'Korea':
        final_texts = extracted_texts['Korea']
    elif best_lang == 'Japan':
        final_texts = extracted_texts['Japan']
    else:
        final_texts = extracted_texts['China']

    # 텍스트가 비어있으면 백업으로 ch 모델 결과 사용
    if not final_texts: 
        final_texts = extracted_texts['China']

    if best_score < 0.8:
        return "No_label", final_texts

    return best_lang, final_texts

# ==========================================
# [함수] YOLO 기능 분류
# ==========================================
def determine_function_with_yolo(model, img):
    results = model(img, verbose=False, conf=0.4)
    found_classes = []
    for result in results:
        for box in result.boxes:
            found_classes.append(int(box.cls[0]))
    
    if CLASS_ID_ICON in found_classes: return "id"
    elif CLASS_BACK_ICON in found_classes: return "back"
    return "unknown"

# ==========================================
# [메인]
# ==========================================
def process_directory(target_dir, readers, yolo_model):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []

    print(f"[{target_dir}] 로딩 중...")
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, file))
    
    image_paths = sorted(image_paths)
    rows = []
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    print(f"-> 총 {len(image_paths)}장 처리 시작")

    for img_path in tqdm(image_paths):
        ground_truth = os.path.basename(os.path.dirname(img_path))
        img = cv2.imread(img_path)
        if img is None: continue

        h, w, _ = img.shape
        y1, y2 = max(0, FIXED_Y_START), min(h, FIXED_Y_END)
        x1, x2 = max(0, FIXED_X_START), min(w, FIXED_X_END)
        crop = img[y1:y2, x1:x2]
        
        processed_crop = preprocess_for_ocr(crop)

        # 수정된 함수 호출
        lang_result, ocr_texts = determine_language_logic(readers, processed_crop)
        
        final_prediction = "No_label"

        if lang_result == "No_label":
            final_prediction = "No_label"
        elif lang_result in ["Korea", "Japan", "Taiwan"]:
            final_prediction = lang_result
        elif lang_result in ["China", "English"]:
            func = determine_function_with_yolo(yolo_model, img)
            if func == "unknown":
                score_id, _ = calculate_match_score(ocr_texts, ['ID', 'Copy', 'Card', '复印', 'ID卡'])
                if score_id >= 0.8: func = "id"
                else: func = "back"
            final_prediction = f"{lang_result}_{func}"

        is_correct = (ground_truth == final_prediction)
        stats[ground_truth]['total'] += 1
        if is_correct: stats[ground_truth]['correct'] += 1

        rows.append({
            "image_path": img_path,
            "ground_truth": ground_truth,
            "predicted": final_prediction,
            "is_correct": "O" if is_correct else "X",
            "lang_detected": lang_result,
            "ocr_texts": " | ".join(ocr_texts)
        })

    return rows, stats

if __name__ == "__main__":
    logging.getLogger("ppocr").setLevel(logging.WARNING)

    print(">>> Loading OCR Models (CPU Mode)...")
    readers = {
        'ch':     PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=False),
        'korean': PaddleOCR(use_angle_cls=False, lang='korean', use_gpu=False),
        'japan':  PaddleOCR(use_angle_cls=False, lang='japan', use_gpu=False)
    }

    yolo_weights = r"/home/shlee/Final_code/Canon_Printer/best.pt"
    print(f">>> Loading YOLO from {yolo_weights}...")
    yolo_model = YOLO(yolo_weights)

    target_dir = r"/home/shlee/Pass"

    if os.path.exists(target_dir):
        # GPU 캐시 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass

        results, total_stats = process_directory(target_dir, readers, yolo_model)
        
        save_csv = r"/home/shlee/Pass_Selected/result_final_v12_perfect_log_all_data.csv"
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "predicted", "is_correct", "lang_detected", "ocr_texts"])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n[완료] 결과 저장됨: {save_csv}")
        
        print("\n" + "="*40)
        print(f"{'Class':<15} | {'Acc (%)'}")
        print("-" * 40)
        total_cnt, correct_cnt = 0, 0
        for label, d in sorted(total_stats.items()):
            t, c = d['total'], d['correct']
            acc = (c/t)*100 if t>0 else 0
            print(f"{label:<15} | {acc:.1f}% ({c}/{t})")
            total_cnt += t
            correct_cnt += c
        print("-" * 40)
        if total_cnt > 0:
            print(f"TOTAL ACCURACY  | {(correct_cnt/total_cnt)*100:.1f}%")
        print("="*40)
    else:
        print(f"경로 없음: {target_dir}")