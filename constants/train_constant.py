import os
import datetime


class TrainConstant:
    LABEL = ['obj6', 'obj7']

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    num_classes = 3  # 物体种类 + 1 (因为包含了背景)

    root_path = "/home/ubuntu/user/zhanghaozheng/Deformable-DETR-main"

    pretrained_model_before_convert_path = os.path.join(root_path, "Deformable-DETR-main-repo", "pretrained_model", "r50_deformable_detr-checkpoint.pth")
    pretrained_model_after_convert_path = os.path.join(root_path, "Deformable-DETR-main-repo", "converted_pretrained_model", "deformable_detr-r50_%d.pth" % num_classes)

    coco_path = os.path.join(root_path, "data", "coco", "coco_data")

    output_path = os.path.join(root_path, "Deformable-DETR-main-repo", "evaluate_result", "class%d" % num_classes, date_time)

    detect_model_path = os.path.join(root_path, "Deformable-DETR-main-repo", "train_result", "class%d" % num_classes, "2024-11-01 19:59:00", "checkpoint0499.pth")
    test_images_path = os.path.join(root_path, "test_images")