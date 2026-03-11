import os
import datetime


class TrainConstant:
    LABEL = ['obj6', 'obj7']

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    num_classes = 3  # 物体种类 + 1 (因为包含了背景)

    root_path = "/root/zhanghaozheng/project/DeformableDetrTest"

    pretrained_model_before_convert_path = os.path.join(root_path, "Deformable-DETR-main-repo", "pretrained_model", "r50_deformable_detr-checkpoint.pth")
    pretrained_model_after_convert_path = os.path.join(root_path, "Deformable-DETR-main-repo", "converted_pretrained_model", "deformable_detr-r50_%d.pth" % num_classes)

    coco_path = os.path.join(root_path, "data", "coco", "coco_data")

    output_path = os.path.join(root_path, "Deformable-DETR-main-repo", "train_result", "class%d" % num_classes, date_time)

    detect_model_path = os.path.join(root_path, "Deformable-DETR-main-repo", "train_result", "class%d" % num_classes, "history", "checkpoint0499.pth")
    test_images_path = os.path.join(root_path, "test_images")

    val_gt_path = os.path.join(coco_path, "annotations", "instances_val2017.json")
    val_img_path = os.path.join(coco_path, "val2017")  # 验证集图片目录
    val_output_path = os.path.join(root_path, "Deformable-DETR-main-repo", "val_result")  # 测试结果
