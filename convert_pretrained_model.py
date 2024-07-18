import torch
from constants.train_constant import TrainConstant

# 加载官方提供的权重文件，修改成自己的路径
pretrained_weights = torch.load(TrainConstant.pretrained_model_before_convert_path)

# 修改相关权重
num_class = TrainConstant.num_classes  # 自己数据集分类数
pretrained_weights['model']['class_embed.0.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.0.bias'].resize_(num_class + 1)
pretrained_weights['model']['class_embed.1.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.1.bias'].resize_(num_class + 1)
pretrained_weights['model']['class_embed.2.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.2.bias'].resize_(num_class + 1)
pretrained_weights['model']['class_embed.3.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.3.bias'].resize_(num_class + 1)
pretrained_weights['model']['class_embed.4.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.4.bias'].resize_(num_class + 1)
pretrained_weights['model']['class_embed.5.weight'].resize_(num_class + 1, 256)
pretrained_weights['model']['class_embed.5.bias'].resize_(num_class + 1)
# 此处50对应生成queries的数量，根据main.py中--num_queries数量修改
pretrained_weights['model']['query_embed.weight'].resize_(50, 512)
torch.save(pretrained_weights, TrainConstant.pretrained_model_after_convert_path)
