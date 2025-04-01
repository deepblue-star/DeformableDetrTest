import json

import cv2
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchcam.utils import overlay_mask
from torch.nn.functional import dropout,linear,softmax
from tqdm import tqdm

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from main import get_args_parser as get_main_args_parser
from models import build_model
from constants.train_constant import TrainConstant
from visualize.AttentionVisualizer import AttentionVisualizer
from pathlib import Path
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from constants.train_constant import TrainConstant
import util.misc as utils


torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))

LABEL = TrainConstant.LABEL

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=True):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))

        print(label_text)

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    # 修改成自己要保存的目录
    if imwrite:
        if not os.path.exists("./output/pred03"):
            os.makedirs('./output/pred03')
        cv2.imwrite('./output/pred03/{}'.format(save_name), opencvImage)


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def load_model(model_path, args):
    model, criterion, postprocessors = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model, criterion, postprocessors


# 图像的推断
def detect(im, model, transform, prob_threshold=0.97):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    # print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, end - start

def visualize_attention_v3(img_path, model, prob_threshold=0.97):
    w = AttentionVisualizer(model)
    result = w.run_and_return_img(img_path, prob_threshold)
    return result

def save_heatmap(viz_result_dic, image, save_name):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    heat_mask = viz_result_dic["enc_attention_visualization"]
    reference_points = viz_result_dic["reference_points"]
    if heat_mask is not None:
        heat_mask = heat_mask[:, :, :3]
        # 步骤 0：h w转成w h
        heat_mask = heat_mask.transpose(1, 0, 2)
        # 步骤 1：归一化到 [0, 255]
        heat_mask = (heat_mask - heat_mask.min()) / (heat_mask.max() - heat_mask.min())  # 将数据归一化到 [0, 1]
        heat_mask = heat_mask * 255  # 映射到 [0, 255]
        # 步骤 2：转换为 uint8 类型
        heat_mask = heat_mask.astype(np.uint8)

        mask_alpha = 0.5
        masked_img = cv2.addWeighted(image, mask_alpha, heat_mask, mask_alpha, 0)
        M_rescale = np.ones((2, 2))
        M_rescale[0][0], M_rescale[1][1] = masked_img.shape[0], masked_img.shape[1]
        for rp in reference_points:
            rp_rescaled = np.matmul(rp, M_rescale)
            cv2.circle(masked_img, rp_rescaled.astype(np.int32), 5, (0, 255, 0), thickness=-1)
    else:
        masked_img = image

    if not os.path.exists("./output/pred03"):
        os.makedirs('./output/pred03')
    split_name = save_name.split(".")
    split_name.insert(-1, "_heatmap.")
    save_name = "".join(split_name)

    cv2.imwrite('./output/pred03/{}'.format(save_name), masked_img)

def save_feature_map(viz_result_dic, save_name):
    feature_map = viz_result_dic["backbone_features_visualization"]

    if not os.path.exists("./output/pred03"):
        os.makedirs('./output/pred03')
    split_name = save_name.split(".")
    split_name.insert(-1, "_feature_map.")
    save_name = "".join(split_name)

    cv2.imwrite('./output/pred03/{}'.format(save_name), feature_map)


def after_detect(scores, boxes):
    """模拟检测函数（需替换为实际检测逻辑）
    返回格式：[ (x1,y1,x2,y2, score, class_id), ... ]"""
    res = []
    for p, (xmin, ymin, xmax, ymax) in zip(scores, boxes):
        cl = p.argmax()
        res.append([xmin, ymin, xmax, ymax, p[cl], cl])
    # 示例检测结果（实际应替换为模型推理代码）
    return np.array(res)


def convert_to_coco_format(detections, image_id):
    """将检测结果转换为COCO格式[2,5](@ref)"""
    results = []
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        results.append({
            "image_id": image_id,
            "category_id": cls_id + 1,  # 假设标注类别从1开始[6](@ref)
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # 转为xywh格式[3](@ref)
            "score": score
        })
    return results


def evaluate_on_coco_val():
    main_args = get_main_args_parser().parse_args()
    # 加载模型 修改成自己路径
    dfdetr, _, _ = load_model(TrainConstant.detect_model_path, main_args)  # <--修改为自己加载模型的路径
    # <--修改为待预测图片所在文件夹路径
    list_path = TrainConstant.test_images_path

    # 创建输出目录
    os.makedirs(TrainConstant.val_output_path, exist_ok=True)

    # 加载验证集标注[4,6](@ref)
    coco_gt = COCO(TrainConstant.val_gt_path)

    predictions = []
    prob_threshold = 0.97

    for img_info in tqdm(coco_gt.dataset['images']):
        img_path = Path(TrainConstant.val_img_path) / img_info['file_name']
        im = Image.open(img_path)
        scores, boxes, waste_time = detect(im, dfdetr, transform, prob_threshold)
        detections = after_detect(scores, boxes)
        predictions.extend(
            convert_to_coco_format(detections, img_info['id'])
        )

    # 保存预测文件[5](@ref)
    pred_path = Path(TrainConstant.val_output_path) / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    # 执行评估[3,5](@ref)
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # 显式启用多面积范围评估
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [
        96 ** 2, 1e5 ** 2]]
    coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']  # 与上述范围一一对应

    # 执行评估流程
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 获取关键指标
    metrics = {
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[5],
        'mAP': coco_eval.stats[0]
    }

    # 可视化PR曲线[3](@ref)
    plt.figure(figsize=(10, 6))
    for i, iou_thr in enumerate([0.5, 0.75]):
        precision = coco_eval.eval['precision'][i, :, 0, 0, 2]
        recall = coco_eval.params.recThrs
        plt.plot(recall, precision, label=f'IoU={iou_thr}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(Path(TrainConstant.val_output_path) / 'pr_curve.png')

    print(f"\n评估结果：\n"
          f"AP@50: {metrics['AP50'] * 100:.1f}%\n"
          f"AP@75: {metrics['AP75'] * 100:.1f}%\n"
          f"mAP@[50:95]: {metrics['mAP'] * 100:.1f}%")


def run_on_coco_val_set():
    evaluate_on_coco_val()


def run_on_select_test_images():

    main_args = get_main_args_parser().parse_args()
    # 加载模型 修改成自己路径
    dfdetr, _, _ = load_model(TrainConstant.detect_model_path, main_args)  # <--修改为自己加载模型的路径
    # <--修改为待预测图片所在文件夹路径
    list_path = TrainConstant.test_images_path
    files = os.listdir(list_path)

    cn = 0
    waste = 0
    prob_threshold = 0.97
    for file in files:
        img_path = os.path.join(list_path, file)
        im = Image.open(img_path)
        scores, boxes, waste_time = detect(im, dfdetr, transform, prob_threshold)
        plot_result(im, scores, boxes, save_name=file, imshow=False, imwrite=True)
        viz_result_dict = visualize_attention_v3(img_path, dfdetr, prob_threshold)
        save_heatmap(viz_result_dict, im, file)
        save_feature_map(viz_result_dict, file)
        print("{} [INFO] {} detect time: {} done!!!".format(cn, file, waste_time))

        cn += 1
        print(cn)
        waste += waste_time
        waste_avg = waste / cn
        print(waste_avg)


def run_by_original_evaluate():
    main_args = get_main_args_parser().parse_args()
    # 加载模型 修改成自己路径
    dfdetr, criterion, postprocessors = load_model(TrainConstant.detect_model_path, main_args)  # <--修改为自己加载模型的路径
    dataset_val = build_dataset(image_set='val', args=main_args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, main_args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=main_args.num_workers,
                                 pin_memory=True)
    base_ds = get_coco_api_from_dataset(dataset_val)
    test_stats, coco_evaluator = evaluate(dfdetr, criterion, postprocessors,
                                          data_loader_val, base_ds, device, main_args.output_dir)



if __name__ == "__main__":
    # 在test_images文件夹中跑，生成热力图等
    # run_on_select_test_images()
    # 用自己写的coco evaluator，不好用
    # run_on_coco_val_set()
    # 用项目自带的coco evaluator，可以用
    run_by_original_evaluate()

