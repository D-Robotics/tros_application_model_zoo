# 本示例使用skimage，如果是opencv/PIL会有所区别
import skimage
import os
import random
from pathlib import Path
import skimage.io
import numpy as np
from pycocotools.coco import COCO
from horizon_tc_ui.data.transformer import (CenterCropTransformer,
                                            HWC2CHWTransformer,
                                            MeanTransformer,
                                            RGB2BGRTransformer,
                                            ScaleTransformer,
                                            PadResizeTransformer,
                                            ResizeTransformer)

def data_transformer():
    transformers = [
        # PadResizeTransformer((224, 224), pad_value=0, pad_position='bottom_right'),
        # RGB2BGRTransformer(),
        # ScaleTransformer(scale_value=255),
        ResizeTransformer(target_size=(224, 224)),
        # 对输入图片中的所有像素值做减去 mean_value
        # MeanTransformer(means=np.array([103.94, 116.78, 123.68])),
        # # 对输入图片中的所有像素值做乘以data_scale系数
        # ScaleTransformer(scale_value=0.017)
    ]

    return transformers

# src_image 标定集中的原图片
# dst_file 存放最终标定样本数据的文件名称
def convert_image(src_image, dst_dir, transformers):
    image = [skimage.img_as_float(
        skimage.io.imread(src_image)).astype(np.float32)]
    if image[0].ndim < 3:
        return
    for trans in transformers:
        image = trans(image)
    image = image[0].astype(np.float32)
    # 以二进制形式存储标定样本到数据文件
    # breakpoint()
    save_path = os.path.join('kps/aircanvas', dst_dir, Path(src_image).stem + '.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, image[np.newaxis, ...])

def list_files(folder_path):
    file_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_list.append(item_path)
    return file_list


if __name__ == '__main__':
    dst_dir = 'calibration_data_bgr_s100/input_1'
    current_dir = './kps/aircanvas/test_images/'
    selected_images = list_files(current_dir)
    transformers = data_transformer()
    for src_image in selected_images:
        convert_image(src_image, dst_dir, transformers)