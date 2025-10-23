import torch
from ultralytics import YOLO

from globals import PILL_DETECTION

def make_model(device):
    #model = YOLO('yolov8m.pt')
    model = YOLO('yolov8l.pt')
    model.to(device)
    return model

def train_model(model, yaml_path, base_dir):
    # 학습 파라미터
    results = model.train(
        data=yaml_path,
        epochs=1,  ##35,  # 최대 20 에폭
        imgsz=800,  # 이미지 크기
        batch=8,  # 배치 크기
        patience=10,  # Early stopping patience (10 에폭 동안 개선 없으면 중단)
        save=True,  # 모델 저장
        device=0 if torch.cuda.is_available() else 'cpu',  # GPU 자동 선택
        project=f'{base_dir}/yolo_runs',  # 결과 저장 폴더
        name=PILL_DETECTION,  ##'pill_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,  # 초기 learning rate
        lrf=0.01,  # 최종 learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,  ##3,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        label_smoothing=0.0,
        val=True,  # Validation 수행
        plots=True,  # 학습 그래프 자동 생성
        verbose=True
    )

    print("\n 학습 완료!")
    print(f" 결과 저장 위치: {base_dir}/yolo_runs/{PILL_DETECTION}")

    return results