import wandb, numpy as np
from ultralytics import YOLO

def train_large():
# wandb.init(
#     project="pill-detection",
#     name="yolos_before_ensemble22",
#     config={
#         "model": "yolov8l.pt",
#         "epochs": 100,
#         "imgsz": 640,
#         "batch": 8,
#         "optimizer": "Adam",
#         # "weight_decay": 0.0,
#         "lr0": 0.00003,
#         "cos_lr": True,
#         "aug": "No",
#         # "patience": 10
#     }
# )




    model = YOLO("yolov8l.pt")






# def on_fit_epoch_end(trainer):
#     try:
#         metrics = trainer.metrics
#         train_losses = trainer.label_loss_items(trainer.tloss, prefix="train")
#
#         wandb.log({
#             "epoch": trainer.epoch,
#             "train/box_loss": train_losses.get("train/box_loss", np.nan),
#             "train/cls_loss": train_losses.get("train/cls_loss", np.nan),
#             "train/dfl_loss": train_losses.get("train/dfl_loss", np.nan),
#             "val/precision": metrics.get("metrics/precision(B)", np.nan),
#             "val/recall": metrics.get("metrics/recall(B)", np.nan),
#             "val/mAP50": metrics.get("metrics/mAP50(B)", np.nan),
#             "val/mAP50-95": metrics.get("metrics/mAP50-95(B)", np.nan),
#         })
#     except Exception as e:
#         print(f"⚠️ W&B 로깅 중 오류: {e}")
#
# model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

# 🔥 fine-tuning 시작
    results = model.train(
        data="../../ai05-level1-project/yolo_dataset/data.yaml",
        epochs=100,
        # patience=10,  # early stop
        cos_lr=True,  # cosine scheduler
        imgsz=640,
        batch=8,
        augment=False,
        optimizer="Adam",
        lr0=0.00003,
        # weight_decay=0.005,
        # momentum=0.937,
        project="../../models/yolo_runs",
        name="yolo_ensemble_large",
        pretrained=True,
        device=0, # GPU 사용시 0으로 변경, Mac local 사용시 "mps"로 변경
        verbose=True,
        # save = True,
    )
    return model