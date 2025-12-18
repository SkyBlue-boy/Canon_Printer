# 🖨️ Canon Printer QA Inspection System

프린터 제조 공정의 최종 품질 검사(QA)를 자동화하기 위한 시스템 (프린터기 불량 검출 및 기종 분류 자동화)

## Pipeline Overview(app.py)

Streamlit 기반의 웹 애플리케이션으로, PatchGuard(이상치 탐지)와 YOLO(객체 탐지), OCR(문자 인식) 기술을 결합하여 불량 검출부터 기종 분류까지의 전 과정을 하나의 파이프라인으로 통합했습니다.
시스템은 app.py를 통해 실행되며, 업로드된 이미지에 대해 다음과 같은 순차적인 검사 및 분류를 수행합니다.

#### 1. 이미지 입력 (Input)
- Streamlit UI를 통해 다수의 테스트 이미지를 업로드합니다.
- 시스템은 이미지를 로드하고 분석을 위한 전처리를 수행합니다.

#### 2. 3단계 결함 검사 (Defect Inspection)
모든 이미지는 먼저 불량 여부를 판별하기 위해 3가지 체크포인트를 거칩니다. 하나라도 통과하지 못하면 즉시 FAIL로 판정됩니다.

| 단계 | 검사 항목 | 사용 모델 / 알고리즘 | 판정 기준 (Threshold) | 시각화 (Visualization) |
| :---: | :--- | :--- | :--- | :--- |
| 1️⃣ **Step 1** | 🖐️ **화면 침범** (Intrusion) | `PatchGuard` (Printer) | Anomaly Score ≥ `0.8` | 🔴 **Red Heatmap** (JET) |
| 2️⃣ **Step 2** | ✨ **빛 반사** (Reflection) | `PatchGuard` (LCD) | Anomaly Score ≥ `-1.0` | 🟠 **Red/Orange Heatmap** (HOT)<br><sub>*LCD 영역 오버레이*</sub> |
| 3️⃣ **Step 3** | 📏 **아이콘 밀림** (Position) | `YOLOv8` | Max Distance > `5.0 px` | 🔗 **아이콘-버튼 중심점 연결선** |

#### 3. 기종 분류 (Model Classification) - PASS Only
결함 검사를 통과(PASS)한 정상 제품에 대해서만 기종 분류가 수행됩니다.

- 알고리즘: PaddleOCR + YOLOv12
- 분류 로직:
  - 언어 식별: OCR을 통해 한국어, 일본어, 중국어(번체/간체) 키워드 매칭.
  - 기능 식별: YOLO 객체 탐지 및 OCR 텍스트를 통해 ID Card / Back 기능 버튼 구분.
- 분류 클래스: Korea, Japan, Taiwan, China_id, China_back, English_id, English_back, No_label


#### 4. 결과 처리 및 데이터 관리 (Output & Logging)
검사 결과는 UI에 실시간으로 표시되며, 로컬 스토리지에 체계적으로 저장됩니다.

- 자동 분류 저장:
  - 📁 FAIL 폴더: 불량 이미지를 저장하며, 파일명에 불량 원인을 태깅합니다. (예: img_FAIL_[Intrusion_Reflection].jpg)
  - 📁 PASS 폴더: 분류된 기종별로 하위 폴더를 생성하여 저장합니다. (예: PASS/Korea/img.jpg)

- 시각화 저장: 불량 이미지의 경우, 불량 위치를 히트맵으로 표시한 _vis.jpg 파일을 함께 생성합니다.
- 로그 기록: 모든 검사 내역(파일명, 결과, 불량 원인, 탐지된 기종, 세부 점수)을 inspection_log.csv 파일로 저장합니다.

📂 Directory Structure (Result)

```bash
Results/Run_YYYYMMDD_HHMMSS/
│
├── 📂 FAIL/                            # 불량품 저장소 (NG)
│   ├── image_01_FAIL_[Intrusion].jpg   # 원본 이미지
│   └── image_01_FAIL_[Intrusion]_vis.jpg # 히트맵 시각화 (Visualization)
│
├── 📂 PASS/                            # 정상품 저장소 (OK)
│   ├── 📂 Korea/                       # 국가/기종별 자동 분류
│   ├── 📂 English_id/
│   └── ...
│
└── 📊 inspection_log.csv               # 전체 검사 결과 및 통계 로그
```

## 핵심 코드 설명

```bash
Canon_Printer(Project_Root folder)/
│
├── app.py  (Streamlit 코드)
├── best.pt (YOLO Icon, Button BBOX 모델 가중치)
├── patchguard_mvtec_printer.pth (PatchGuard Printer 가중치)
├── patchguard_mvtec_lcd.pth (PatchGuard LCD 가중치)
├── EDSR_x4.pb (Super Resolution 모델, 선택사항)
│
├── anomaly/
│   ├── PatchGuard/
│   │   ├── patchguard.py
│   │   ├── utils.py
│   │   └── ... (기타 의존성 파일)
│   └── yolo_inference_icon_position.py (참고용, 로직은 app.py에 통합됨)
│
└── printer_classification/
    └── yolo+ocr_sr_mis_data_add_final.py (참고용, 로직은 app.py에 통합됨)
```

### PatchGuard 
- 화면 침범(ex. hand, 정상 프린터의 생김새가 아닌 여러 엣지 케이스 이미지, etc.)이랑 lcd 판의 빛 반사, 비닐 구겨짐을 이상치로 잡는 코드
- patchguard_mvtec_printer.pth: 배경 없이 프린터 부분만 크롭하여 정상 데이터 학습
- patchguard_mvtec_lcd.pth: lcd 패널 화면 부분만 크롭하여 빛 반사, 비닐 구겨짐 없는 정상 데이터 학습 

### yolo_inference_icon_position.py 
1. 결과 파싱: YOLO 결과(result.boxes)에서 각 클래스별 BBox 좌표를 추출합니다.
2. 중심점 계산: $(x1, y1, x2, y2)$ 좌표를 이용해 중심점 $(cx, cy)$를 구합니다.
3. 거리 계산: 매칭되는 쌍(예: Home_button <-> Home_icon)의 중심점 간 유클리드 거리(Euclidean Distance)를 계산합니다.
4. Pass/Fail 판정: 미리 설정한 허용 오차(Threshold)보다 거리가 크면 Fail, 작으면 Pass로 처리합니다.

### yolo+ocr_sr_mis_data_add_final.py
- YOLOv12와 Paddle_OCR 모델을 활용해서 프린터의 8개 기종을 분류하는 코드


## Training PatchGuard

- printer class(손 침범 등) 학습을 할 때는 --class_name에 printer, lcd class(빛 반사, 화면 비닐 구겨짐 등) 학습을 할 때는 --class_name에 lcd

- Train: python main.py --mode train --class_name printer --dataset mvtec --dataset_dir /home/shlee/Final_code/Canon_Printer/anomaly/datasets/all --epochs 100
- Test: python main.py --mode test --class_name printer --dataset mvtec --dataset_dir /home/shlee/Final_code/Canon_Printer/anomaly/datasets/all --checkpoint_dir /home/shlee/Final_code/Canon_Printer/anomaly/PatchGuard/
- Visualization: python main.py --mode visualization --class_name printer --dataset mvtec --dataset_dir /home/shlee/Final_code/Canon_Printer/anomaly/datasets/all --checkpoint_dir /home/shlee/Final_code/Canon_Printer/anomaly/PatchGuard/

- PatchGuard/patchguard_mvtec_printer.pth, PatchGuard/patchguard_mvtec_lcd.pth 이런식으로 학습된 가중치 파일이 PatchGuard 폴더 안에 존재 해야 함

## 💻 Tech Stack
- Framework: Streamlit
- Anomaly Detection: PatchGuard (Vision Transformer based)
- Object Detection: YOLOv12 (Ultralytics)
- OCR: PaddleOCR
- Image Processing: OpenCV, SciPy (Gaussian Filter)

## Demo
- Demo by Streamlit
- 의존성 설치 (NumPy 버전 호환성 주의): pip install streamlit opencv-python torch torchvision pandas scipy ultralytics paddlepaddle paddleocr timm "numpy<2.0"
- paddleocr_Version: 2.8.1, paddlepaddle_Version: 3.0.0

- 🛠️ How to Run: cd ~/Final_code/Canon_Printer --> streamlit run app.py

[streamlit-app-demo.webm](https://github.com/user-attachments/assets/e903af2e-6640-4543-b55a-60a7076582f8)




