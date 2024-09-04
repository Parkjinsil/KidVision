
import numpy as np
import cv2
import os

#에지 검출 : 흑백 -> 가우시안블러링 -> 캐니
def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 123, 255, cv2.THRESH_TOZERO)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=15, sigma_r=0.2)

    edged = cv2.Canny(gray, 75, 200, True)
    return edged

def contours(edge, path):
    cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 모든 윤곽선의 점들을 합쳐서 하나의 윤곽선처럼 다룰 수 있도록 점 리스트를 합침
    all_points = np.vstack(cnts)

    # 윤곽선의 시작점과 끝점을 연결하여 윤곽선을 폐합
    if len(all_points) > 0:
        all_points = np.vstack([all_points, [all_points[0]]])

    # 마스크 생성
    mask = np.zeros(resize_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [all_points], -1, 255, thickness=cv2.FILLED)

    # 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    # 직사각형에 맞춰 이미지를 자르기
    cropped = resize_img[y:y+h, x:x+w]

    # 자른 이미지를 512x512 크기로 리사이즈
    resized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, resized)

def write_img(filename, path):
    global resize_img
    ori_img = cv2.imread(filename)
    ori_img = ori_img[660:, :]
    resize_img = cv2.resize(ori_img, dsize=(512, 512), interpolation=cv2.INTER_AREA)  

    edged = edge_detection(resize_img)

    contours(edged, path)

def canvas_detect(path, output_dir):
    cats = ['house', 'person', 'tree']
    for cat in cats:
        filenames = os.listdir(path)
        for i in range(len(filenames)):
            if cat in filenames[i]:
                filename = os.path.join(path, filenames[i])
                join_path = os.path.join(output_dir , cat)
                join_path = os.path.join(join_path, filenames[i])
                write_img(filename, join_path)