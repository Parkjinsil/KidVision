import sys
sys.path.append("../")
sys.path.append("./detr")
sys.path.append("./detr/detr")

import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import cv2
import copy

import torch
from detr_model import DETRModel

NAMES = {"house": {
            'Door' : 0,
            'House' :1, 
            'Roof' : 2, 
            'Window' : 3},
        "tree" : {
            'Branch' : 0, 
            'Crown' : 1, 
            'Fruit' : 2, 
            'Gnarl' : 3, 
            'Root' : 4, 
            'Tree' : 5},
        "person" : {
            'Arm' : 0, 
            'Eye' : 1, 
            'Leg' : 2, 
            'Mouth': 3, 
            'Person' : 4
        }
}
# DETR 모델의 출력값을 이용하여 평균 박스 좌표를 계산하는 함수입니다.
def get_average_box_coordinates(output, image_width, image_height):
    boxes = output['pred_boxes'][0].detach().numpy()
    boxes[:, 0] *= image_width
    boxes[:, 1] *= image_height
    boxes[:, 2] *= image_width
    boxes[:, 3] *= image_height

    average_box = np.mean(boxes, axis=0)
    (x1, y1, x2, y2) = tuple(average_box)
    width = x2 - x1
    height = y2 - y1
    area = width * height
    arean = area / (image_width * image_height)
    xcenter = x1 + width / 2
    xcentern = xcenter / image_width
    return arean, xcentern

def handling_yolos(classes, images_path, csv_folder_path):
    house_model = YOLO('yolos/house/house400_best.pt')
    house_model2 = YOLO('yolos/house/house_best.pt')
    csv_names = os.listdir(csv_folder_path)
    house_csv_name = [name for name in csv_names if "house.csv" in name]
    house_csv_path = os.path.join(csv_folder_path, house_csv_name[0])
    house_csv = pd.read_csv(house_csv_path)

    tree_model = YOLO('models/tree_best.pt')
    tree_csv_name = [name for name in csv_names if "tree.csv" in name]
    tree_csv_path = os.path.join(csv_folder_path, tree_csv_name[0])
    tree_csv = pd.read_csv(tree_csv_path)

    person_model = YOLO('models/person_best.pt')
    person_csv_name = [name for name in csv_names if "person.csv" in name]
    person_csv_path = os.path.join(csv_folder_path, person_csv_name[0])
    person_csv = pd.read_csv(person_csv_path)

    detr_model = DETRModel(num_classes=5, num_queries=100)
    detr_model.load_state_dict(torch.load("./models/detr_best.pth"))

    for cat in classes:
        filenames = os.listdir(images_path[cat])
        filenames = [os.path.join(images_path[cat], filename) for filename in filenames]
        if cat == "house":
            model = house_model
            model2 = house_model2
            result_house_csv, house_except_list = house_predict_to_csv(model, model2, detr_model, filenames, house_csv)
            result_house_csv.to_csv("./test_house.csv", index=False)
            print(house_except_list)
        if cat == "tree":
            model = tree_model
            result_tree_csv, tree_except_list = tree_predict_to_csv(model, detr_model, filenames, tree_csv)
            result_tree_csv.to_csv("./test_tree.csv", index=False)
            print(tree_except_list)
        if cat == "person":
            model = person_model
            result_person_csv, person_except_list = person_predict_to_csv(model, detr_model, filenames, person_csv)
            result_person_csv.to_csv("./test_person.csv", index=False)
            print(person_except_list)


def house_predict_to_csv(model, model2, detr_model, filenames, csv):
    names = NAMES["house"]
    except_images = []
    # 이미지별 예측 및 정확도 계산
    for index, filename in enumerate(filenames):
        results = model.predict(filename)
        results2 = model2.predict(filename)
        file_id = results[0].path.split('/')[-1].split('.')[0]
        print(f"id = {file_id}, predicted")
        class_boxes = []
        window_cnt = 0
        df_index = {
            "id": file_id,
            "door_yn": "n",
            "roof_yn": "n"}
        for box in results[0].boxes:
            box_data = {
                'cls': box.cls,
                'conf': box.conf,
                'xywh': box.xywh,
                'area': box.xywh[0][2] * box.xywh[0][3] / (512 * 512),
                'x_center_point': box.xywh[0][0] / 512,
                'y_center_point': box.xywh[0][1] / 512
            }
            if box_data['cls'][0] == names["Door"]:
                df_index['door_yn'] = "y"
            elif box_data['cls'][0] == names["Roof"]:
                df_index['roof_yn'] = "y"
            elif box_data['cls'][0] == names["Window"]:
                window_cnt += 1
        for box in results2[0].boxes:
            box_data2 = {
                'cls': box.cls,
                'conf': box.conf,
                'xywh': box.xywh,
                'area': box.xywh[0][2] * box.xywh[0][3] / (512 * 512),
                'x_center_point': box.xywh[0][0] / 512,
                'y_center_point': box.xywh[0][1] / 512
            }
            if box_data2['cls'][0] == names["House"]:
                class_boxes.append(box_data2)
        try:
            class_box = max(class_boxes, key=lambda x: x['area'])
            if class_box["area"] >= 0.53: # 기존 0.6
                df_index["size"] = "big"
            elif class_box["area"] >= 0.14: # 기존 0.16
                df_index["size"] = "middle"
            else:
                df_index["size"] = "small"

            if class_box["x_center_point"] < 1 / 3:
                df_index["loc"] = "left"
            elif class_box["x_center_point"] < 2 / 3:
                df_index["loc"] = "center"
            else:
                df_index["loc"] = "right"
        except:
            try:
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output = detr_model(image)
                arean, locn = get_average_box_coordinates(output, 512, 512)
                if arean >= 0.6:
                    df_index["size"] = "big"
                elif arean >= 0.16:
                    df_index["size"] = "middle"
                else:
                    df_index["size"] = "small"

                if locn < 1 / 3:
                    df_index["loc"] = "left"
                elif locn < 2 / 3:
                    df_index["loc"] = "center"
                else:
                    df_index["loc"] = "right"
                except_images.append(file_id)
            except:
                df_index["size"] = "middle"
                df_index["loc"] = "center"
                except_images.append(file_id)

        if window_cnt >= 3:
            df_index["window_cnt"] = "more than 3"
        elif window_cnt >= 1:
            df_index["window_cnt"] = "1 or 2"
        else:
            df_index["window_cnt"] = "absence"
        
        mask = csv['id'] == df_index['id']
        for key, value in df_index.items():
            if key != 'id':  # 'id' 열은 업데이트 대상이 아니므로 제외
                csv.loc[mask, key] = value
    return csv, except_images

def tree_predict_to_csv(model, detr_model, filenames, csv):
    names = NAMES["tree"]
    except_images = []
    # 이미지별 예측 및 정확도 계산
    for index, filename in enumerate(filenames):
        results = model.predict(filename)
        file_id = results[0].path.split('/')[-1].split('.')[0]
        print(f"id = {file_id}, predicted")
        class_boxes = []
        window_cnt = 0
        df_index = {
            "id": file_id,
            "branch_yn": "n",
            "crown_yn": "n",
            "fruit_yn": "n",
            "gnarl_yn": "n",
            "root_yn": "n"}
        for box in results[0].boxes:
            box_data = {
                'cls': box.cls,
                'conf': box.conf,
                'xywh': box.xywh,
                'area': box.xywh[0][2] * box.xywh[0][3] / (512 * 512),
                'x_center_point': box.xywh[0][0] / 512,
                'y_center_point': box.xywh[0][1] / 512
            }
            if box_data['cls'][0] == names["Tree"]:
                class_boxes.append(box_data)
            elif box_data['cls'][0] == names["Branch"]:
                df_index['branch_yn'] = "y"
            elif box_data['cls'][0] == names["Crown"]:
                df_index['crown_yn'] = "y"
            elif box_data['cls'][0] == names["Fruit"]:
                df_index['fruit_yn'] = "y"
            elif box_data['cls'][0] == names["Gnarl"]:
                df_index['gnarl_yn'] = "y"
            else:
                df_index['root_yn'] = "y"
        try:
            class_box = max(class_boxes, key=lambda x: x['area'])
            if class_box["area"] >= 0.55: # 기존 0.6
                df_index["size"] = "big"
            elif class_box["area"] >= 0.2: # 기존 0.16
                df_index["size"] = "middle"
            else:
                df_index["size"] = "small"

            if class_box["x_center_point"] < 1 / 3:
                df_index["loc"] = "left"
            elif class_box["x_center_point"] < 2 / 3:
                df_index["loc"] = "center"
            else:
                df_index["loc"] = "right"
        except:
            try:
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output = detr_model(image)
                arean, locn = get_average_box_coordinates(output, 512, 512)
                if arean >= 0.6:
                    df_index["size"] = "big"
                elif arean >= 0.16:
                    df_index["size"] = "middle"
                else:
                    df_index["size"] = "small"

                if locn < 1 / 3:
                    df_index["loc"] = "left"
                elif locn < 2 / 3:
                    df_index["loc"] = "center"
                else:
                    df_index["loc"] = "right"
                except_images.append(file_id)
            except:
                df_index["size"] = "middle"
                df_index["loc"] = "center"
                except_images.append(file_id)
        
        mask = csv['id'] == df_index['id']
        for key, value in df_index.items():
            if key != 'id':  # 'id' 열은 업데이트 대상이 아니므로 제외
                csv.loc[mask, key] = value
    return csv, except_images

def person_predict_to_csv(model, detr_model, filenames, csv):
    names = NAMES["person"]
    except_images = []
    # 이미지별 예측 및 정확도 계산
    for index, filename in enumerate(filenames):
        results = model.predict(filename)
        file_id = results[0].path.split('/')[-1].split('.')[0]
        print(f"id = {file_id}, predicted")
        class_boxes = []
        df_index = {
            "id": file_id,
            "arm_yn": "n",
            "eye_yn": "n",
            "leg_yn": "n",
            "mouth_yn": "n"}
        for box in results[0].boxes:
            box_data = {
                'cls': box.cls,
                'conf': box.conf,
                'xywh': box.xywh,
                'area': box.xywh[0][2] * box.xywh[0][3] / (512 * 512),
                'x_center_point': box.xywh[0][0] / 512,
                'y_center_point': box.xywh[0][1] / 512
            }
            if box_data['cls'][0] == names["Person"]:
                class_boxes.append(box_data)
            elif box_data['cls'][0] == names["Arm"]:
                df_index['arm_yn'] = "y"
            elif box_data['cls'][0] == names["Eye"]:
                df_index['eye_yn'] = "y"
            elif box_data['cls'][0] == names["Leg"]:
                df_index['leg_yn'] = "y"
            elif box_data['cls'][0] == names["Mouth"]:
                df_index['mouth_yn'] = "y"
        try:
            class_box = max(class_boxes, key=lambda x: x['area'])
            if class_box["area"] >= 0.58: # 기존 0.6
                df_index["size"] = "big"
            elif class_box["area"] >= 0.16: # 기존 0.16
                df_index["size"] = "middle"
            else:
                df_index["size"] = "small"

            if class_box["x_center_point"] < 1 / 3:
                df_index["loc"] = "left"
            elif class_box["x_center_point"] < 2 / 3:
                df_index["loc"] = "center"
            else:
                df_index["loc"] = "right"
        except:
            try:
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output = detr_model(image)
                arean, locn = get_average_box_coordinates(output, 512, 512)
                if arean >= 0.6:
                    df_index["size"] = "big"
                elif arean >= 0.16:
                    df_index["size"] = "middle"
                else:
                    df_index["size"] = "small"

                if locn < 1 / 3:
                    df_index["loc"] = "left"
                elif locn < 2 / 3:
                    df_index["loc"] = "center"
                else:
                    df_index["loc"] = "right"
                except_images.append(file_id)
            except:
                df_index["size"] = "middle"
                df_index["loc"] = "center"
                except_images.append(file_id)

        mask = csv['id'] == df_index['id']
        for key, value in df_index.items():
            if key != 'id':  # 'id' 열은 업데이트 대상이 아니므로 제외
                csv.loc[mask, key] = value
    return csv, except_images

# 성능평가 이전에 정확도 계산할 때 사용했던 함수입니다.
def get_accuracy():
    house_model = YOLO('yolos/house/house_best.pt')
    house400_model = YOLO('yolos/house/house400_best.pt')
    tree_model = YOLO('yolos/tree/tree400_best.pt')
    person_model = YOLO('yolos/person/person800_best.pt')
    detr_model = DETRModel(num_classes=5, num_queries=100)
    detr_model.load_state_dict(torch.load("./models/detr_best.pth"))

    ####
    model = person_model
    # model400 = house400_model
    ####
    true_csv_path = 'data/noannotated_data/train_person.csv'
    ####
    filenames = os.listdir("/home/connet/jay_deb/DMC-Connet-Team2-Project-1/data/noannotated_data/person")
    ####
    filenames = [os.path.join("/home/connet/jay_deb/DMC-Connet-Team2-Project-1/data/noannotated_data/person", filename) for filename in filenames]
    # CSV 파일 불러오기
    true_df = pd.read_csv(true_csv_path)
    csv = copy.deepcopy(true_df)
    ####
    predicted_df, except_images = person_predict_to_csv(model, detr_model,filenames, csv)
    

    # 컬럼 이름들
    house_columns = ["door_yn", "loc", "roof_yn", "window_cnt", "size"]
    tree_columns = ['branch_yn', 'root_yn', 'crown_yn', 'fruit_yn', 'gnarl_yn', 'loc', 'size']
    person_columns = ['eye_yn', 'leg_yn', 'loc', 'mouth_yn', 'size', 'arm_yn']
    ####
    columns = person_columns
    # 틀린 예측 수 초기화
    total_predictions = {col: 0 for col in columns}
    wrong_predictions = {col: 0 for col in columns}
    total_predictions_count = 0
    wrong_predictions_count = 0

    # predicted_df['id'] = predicted_df['id'].str.replace('.jpg', '')

    # ID를 기준으로 두 데이터프레임을 병합하여 비교
    merged_df = pd.merge(predicted_df, true_df, on='id', suffixes=('_pred', '_true'))

    # 각 컬럼에 대해 정확도 계산
    for col in columns:
        total_predictions[col] = len(merged_df)
        wrong_predictions[col] = (merged_df[f"{col}_pred"] != merged_df[f"{col}_true"]).sum()
        total_predictions_count += len(merged_df)
        wrong_predictions_count += wrong_predictions[col]

    # 컬럼별 정확도 계산
    accuracy = {col: (total_predictions[col] - wrong_predictions[col]) / total_predictions[col] for col in columns}
    accuracy_percentage = {col: accuracy[col] * 100 for col in accuracy}

    print("컬럼별 정확도:")
    for col in accuracy_percentage:
        print(f"{col}: {accuracy_percentage[col]:.2f}%")

    # 전체 정확도 계산
    overall_accuracy = (total_predictions_count - wrong_predictions_count) / total_predictions_count
    overall_accuracy_percentage = overall_accuracy * 100

    print(f"전체 정확도: {overall_accuracy_percentage:.2f}%")
    print(F"예외 이미지 수: {len(except_images)}")
