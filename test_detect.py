import cv2
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torchcam.utils import overlay_mask
from torch.nn.functional import dropout,linear,softmax
from main import get_args_parser as get_main_args_parser
from models import build_model
from constants.train_constant import TrainConstant
from visualize.AttentionVisualizer import AttentionVisualizer


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
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model


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


if __name__ == "__main__":

    main_args = get_main_args_parser().parse_args()
    # 加载模型 修改成自己路径
    dfdetr = load_model(TrainConstant.detect_model_path, main_args)  # <--修改为自己加载模型的路径
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
