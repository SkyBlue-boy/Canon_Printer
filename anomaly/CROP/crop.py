import cv2
import os

# ==========================================
# [설정 1] 이미지 파일 1개 경로 (테스트용)
# ==========================================
TEST_IMG_PATH = "/home/shlee/Pass/Taiwan/1118(1).jpeg"  # 테스트할 이미지 파일 경로

# ==========================================
# [설정 2] 자르고 싶은 영역 좌표 (직사각형)
# ==========================================
# 포토샵이나 그림판에서 확인한 픽셀 좌표를 넣으세요.

## printer 전체 ##

X_START = 300   # x1
Y_START = 200   # y1

# 오른쪽 아래 (End)
X_END   = 1280   # x2
Y_END   = 710   # y2

# # icon ##
# X_START = 310   # x1
# Y_START = 270   # y1

# # 오른쪽 아래 (End)
# X_END   = 700   # x2
# Y_END   = 660   # y2

# ## lcd ##
# # 왼쪽 위 (Start)
# X_START = 730   # x1
# Y_START = 240   # y1

# # 오른쪽 아래 (End)
# X_END   = 1280   # x2
# Y_END   = 620   # y2

def check_roi():
    if not os.path.exists(TEST_IMG_PATH):
        print("이미지 파일이 없습니다. 경로를 확인해주세요.")
        return

    img = cv2.imread(TEST_IMG_PATH)
    
    # 빨간색 네모 그리기 (이미지, 시작좌표, 끝좌표, 색상(BGR), 두께)
    cv2.rectangle(img, (X_START, Y_START), (X_END, Y_END), (0, 0, 255), 3)
    
    # 확인용 이미지 저장
    save_path = "C:/Users/pyjun/Documents/anomaly/CROP/roi_check_result_printer_part13.jpg"
    cv2.imwrite(save_path, img)
    print(f"확인용 이미지가 저장되었습니다: {save_path}")
    print("저장된 이미지를 열어서 빨간 박스가 원하는 위치인지 확인하세요.")

if __name__ == "__main__":
    check_roi()