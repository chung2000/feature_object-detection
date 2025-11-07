import os, torch
from tqdm import tqdm

from src.main.ensemble_wbf import ensemble_wbf
from src.main.make_csv import make_csv
from src.YOLO.convert_data import convert_data
from src.YOLO.make_yaml import make_yaml
from src.YOLO.make_yolo_dir import make_yolo_dir
from src.datas.data_stratify import data_stratify
from src.datas.data_loader import data_loader
from src.main.make_dataframe import search_data
from src.main.train_summary import train_summary
from src.utils.check_json import check_json
from src.datas.transforms import transforms
from src.utils.change_bbox import change_bbox

from src.main.train_large import train_large
from src.main.train_medium import train_medium

from src.utils.process_annotation import process_annotation
from src.utils.korean import set_korean_font
import globals

# 데이터 기본 경로 (압축 해제한 위치)
BASE_DIR = "../../ai05-level1-project"
JSON_PATH = f"{BASE_DIR}/train_combined.json"

# 학습 및 테스트 데이터 경로
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
TRAIN_ANN_DIR = f"{BASE_DIR}/train_annotations"
TEST_IMG_DIR = f"{BASE_DIR}/test_images"

YOLO_DIR = f"{BASE_DIR}/yolo_dataset"

def main():
    # 한글 폰트 설정
    set_korean_font()

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 경로 설정
    check_datapath()

    # Annotation files -> 하나의 json으로 묶기
    train_data, all_json_files = process_annotation(TRAIN_ANN_DIR)

    # iou 0.1 이상, bbox coordinate> img_size들의  Bbox 변경
    change_bbox()


    # 데이터 탐색
    images_df, categories_df, annotations_df = search_data(train_data)

    ### FONT ###
    #set_font()
    #add_font()

    # annotation과
    check_json(all_json_files)

    # 트랜스폼 생성
    train_transform, val_transform = transforms()


    # train, val Stratify
    train_dataset, val_dataset, train_images_df, val_images_df, train_annotations_df, val_annotations_df\
        = data_stratify(images_df, annotations_df, categories_df, train_transform, val_transform, TRAIN_IMG_DIR)


    # data_loader, yolo 내장 dataLoader 사용해서 안 쓸듯
    train_loader, val_loader = data_loader(train_dataset,val_dataset)


    # make yolo dir
    category_id_mapping, num_classes = make_yolo_dir(categories_df)


    convert_data(train_images_df, train_annotations_df, val_images_df, val_annotations_df, category_id_mapping)

    # make_yaml
    make_yaml(categories_df)


    # train
    ### 풀면 미친듯이 학습을 시작할 것임 ###
    model_large = train_large()
    model_medium = train_medium()

    #train summary
    train_summary(categories_df, annotations_df, best_model_path="../../models/L-best.pt")
    train_summary(categories_df, annotations_df, best_model_path="../../models/M-best.pt")

    test_images = sorted(os.listdir(TEST_IMG_DIR))

    # 모든 test 이미지에 대해 예측 수행
    predictions = {}
    for img_name in tqdm(test_images, desc="🔍 앙상블 추론 중"):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        boxes, scores, labels = ensemble_wbf(img_path, conf=0.1, iou_thr=0.55)
        predictions[img_name] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
    print(f"총 {len(predictions)}개의 결과 저장됨.")


    # Kaggle 제출용 CSV 파일 만들기
    make_csv(predictions, category_id_mapping)

    return


def check_datapath():
    # 실제 폴더 및 파일 존재 여부 확인
    print("📂 경로 설정:")
    for name, path in [("BASE_DIR", BASE_DIR),
                       ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
                       ("TRAIN_ANN_DIR", TRAIN_ANN_DIR),
                       ("TEST_IMG_DIR", TEST_IMG_DIR)]:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")


if __name__ == "__main__":
    main()