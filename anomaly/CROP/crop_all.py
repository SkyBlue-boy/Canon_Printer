import cv2
import os
from tqdm import tqdm  # 진행률 표시 (없으면 pip install tqdm)

# ==========================================
# [설정] 경로 및 좌표
# ==========================================
SRC_DIR = "/home/shlee/Final_code/Canon_Printer/anomaly/CROP/printer"       # 원본 이미지가 있는 폴더
DST_DIR = "/home/shlee/Final_code/Canon_Printer/anomaly/CROP/printer_crop"    # 자른 이미지를 저장할 폴더

# # 위에서 확인한 확정 좌표
# X_START = 310   # 왼쪽 (Left)
# Y_START = 220   # 위 (Top)
# X_END   = 780   # 오른쪽 (Right)
# Y_END   = 660   # 아래 (Bottom)

## printer 전체 ##

X_START = 300   # x1
Y_START = 200   # y1

# 오른쪽 아래 (End)
X_END   = 1280   # x2
Y_END   = 710   # y2

# ## printer 전체 - china_id ## 이렇게 해도 위에랑 큰차이는 없어서 256으로 리사이즈 하고 학습하면 ㄱㅊ을듯. 제미나이 왈.

# X_START = 220   # x1
# Y_START = 200   # y1

# # 오른쪽 아래 (End)
# X_END   = 1280   # x2
# Y_END   = 710   # y2

def crop_all_images():
    # 저장 폴더가 없으면 생성
    os.makedirs(DST_DIR, exist_ok=True)

    # 이미지 파일 리스트 가져오기
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(valid_ext)]

    print(f"총 {len(file_list)}장의 이미지를 처리합니다...")

    success_count = 0

    for filename in tqdm(file_list):
        src_path = os.path.join(SRC_DIR, filename)
        dst_path = os.path.join(DST_DIR, filename) # 같은 이름으로 저장

        img = cv2.imread(src_path)
        
        if img is None:
            print(f"[Skip] 이미지 로드 실패: {filename}")
            continue

        # [중요] 이미지 자르기 (Slicing)
        # numpy 배열은 [세로(y):세로(y), 가로(x):가로(x)] 순서입니다.
        cropped_img = img[Y_START:Y_END, X_START:X_END]

        # 예외처리: 좌표가 이미지 크기를 벗어난 경우
        if cropped_img.size == 0:
            print(f"[Error] 좌표 범위 오류 (이미지보다 큼): {filename}")
            continue

        cv2.imwrite(dst_path, cropped_img)
        success_count += 1

    print(f"\n완료! 총 {success_count}장 저장됨.")
    print(f"저장 위치: {DST_DIR}")

if __name__ == "__main__":
    crop_all_images()