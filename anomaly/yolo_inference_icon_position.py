import math
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("/home/shlee/Final_code/Canon_Printer/best.pt")

# 2. 이미지 추론
results = model(["/home/shlee/datasets/eng_back/printer/test/icon_position/732(1).jpeg"])

# 3. 설정 값 정의
PASS_THRESHOLD = 5.0  # 허용 오차 거리 (픽셀 단위). 이 값보다 거리가 멀면 FAIL 처리 (상황에 맞게 조절 필요)

# 검사할 쌍 정의 (Button 클래스명, Icon 클래스명)
check_pairs = [
    ("Home_button", "Home_icon"),
    ("Status_button", "Status_icon"),
    ("Back_button", "Back_icon"),
    ("id_button", "id_icon")
]

def get_center(box):
    """BBox의 중심좌표 (cx, cy)를 반환"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def calculate_distance(p1, p2):
    """두 점 사이의 유클리드 거리 계산"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 4. 결과 처리 루프
for result in results:
    # 시각화를 위해 원본 이미지를 numpy array로 가져옴
    img_vis = result.orig_img.copy()
    
    # 현재 이미지에서 검출된 객체들을 클래스 이름별로 정리
    detected_objects = {}
    
    boxes = result.boxes
    names = result.names
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy() # GPU 텐서를 numpy로 변환
        
        # 동일 클래스가 여러 개일 경우를 대비해 리스트로 저장
        if cls_name not in detected_objects:
            detected_objects[cls_name] = []
        detected_objects[cls_name].append(xyxy)

    # 5. 각 쌍에 대해 거리 계산 및 판정
    print(f"\n--- 분석 결과 ({result.path}) ---")
    
    for btn_name, icon_name in check_pairs:
        # 두 클래스가 모두 검출되었는지 확인
        if btn_name in detected_objects and icon_name in detected_objects:
            # 편의상 가장 신뢰도가 높거나 첫 번째로 잡힌 객체끼리 비교 (여러 개일 경우 로직 추가 필요)
            btn_box = detected_objects[btn_name][0]
            icon_box = detected_objects[icon_name][0]
            
            # 중심점 계산
            btn_center = get_center(btn_box)
            icon_center = get_center(icon_box)
            
            # 거리 계산
            dist = calculate_distance(btn_center, icon_center)
            
            # Pass/Fail 판정
            status = "PASS" if dist <= PASS_THRESHOLD else "FAIL"
            color = (0, 255, 0) if status == "PASS" else (0, 0, 255) # 녹색(PASS), 적색(FAIL)
            
            # 콘솔 출력
            print(f"[{btn_name} <-> {icon_name}] 거리: {dist:.2f} px -> {status}")
            
            # --- 시각화 (이미지에 그리기) ---
            # 1. 중심점 그리기
            cv2.circle(img_vis, (int(btn_center[0]), int(btn_center[1])), 3, (255, 0, 0), -1)
            cv2.circle(img_vis, (int(icon_center[0]), int(icon_center[1])), 3, (0, 0, 255), -1)
            
            # 2. 두 점 잇는 선 그리기
            cv2.line(img_vis, (int(btn_center[0]), int(btn_center[1])), 
                     (int(icon_center[0]), int(icon_center[1])), color, 2)
            
            # 3. 텍스트 표시 (거리 및 상태)
            text = f"{status} ({dist:.1f}px)"
            cv2.putText(img_vis, text, (int(btn_center[0]), int(btn_center[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            print(f"[{btn_name} <-> {icon_name}] : 객체를 찾을 수 없음 (Detection Failed)")

    # 6. 결과 저장 및 보기
    # result.save("result_original.jpg") # YOLO 기본 저장
    cv2.imwrite("/home/shlee/yj/icon_position_inference_image/result_measure_analysis.jpg", img_vis) # 거리 계산이 그려진 이미지 저장
    
    # 윈도우 환경이거나 GUI 가능한 경우만 실행
    # cv2.imshow("Analysis", img_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()