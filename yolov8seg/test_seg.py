#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
# Author: WuChao D-Robotics
# Date: 2024-04-26
# Description: Serial Programming with args, YOLOv8 Seg

import sys
sys.path.append("./python/data/")
from transformer import *
from dataloader import *
from horizon_tc_ui import HB_ONNXRuntime
import numpy as np
import cv2

from postprocess_seg import nms

import cv2
import numpy as np
from scipy.special import softmax
import argparse
from time import time

def yuv_to_rgb(yuv):
    W = np.array([
        [0.00392157, 0, 0.00549686],
        [0.00392157, -0.00134784, -0.00280078],
        [0.00392157, 0.00694980, 0]
    ])
    b = np.array([0.50196081, 0.50196075, 0.50196087])
    return np.dot(W, yuv) + b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='../yolov8n_seg_output/yolov8n_seg_post_640x640_nv12_original_float_model.onnx', help='Path to Horizon BPU Quantized *.bin Model.\nxj3 (bernoulli2) or xj5 (bayes)')
    parser.add_argument('--test-img', type=str, default='./resized_test.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--mask-num', type=int, default=32, help='Mask Num0.')
    parser.add_argument('--input-size', type=int, default=640, help='Model Input Size')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    opt = parser.parse_args()

    # 推理实例
    model = YOLOv8_Seg_img(opt)

    # 读图
    # begin_time = time()
    img = cv2.imread(opt.test_img)
    # print("\033[0;31;40m" + "Read image time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

    # 存储拉伸量
    img_h, img_w = img.shape[0:2]
    print('input_image_size: ', img.shape)
    y_scale, x_scale = img_h/model.input_image_size, img_w/model.input_image_size
    # 这里的160是proto的大小
    y_scale_corp, x_scale_corp = 160/model.input_image_size, 160/model.input_image_size

    # 推理
    begin_time = time()
    # output_tensors = model.forward(opt.test_img)
    dbboxes, scores, ids, masks, proto, indices = model.forward(opt.test_img)
    print("\033[0;31;40m" + "Forward time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

    # # 后处理
    # begin_time = time()
    # output_tensors = getquantioutput()
    # dbboxes, scores, ids, masks, proto, indices = model.postprocess(output_tensors)
    # print("\033[0;31;40m" + "Post Process time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

    # 绘制
    begin_time = time()
    img_h, img_w = img.shape[0:2]

    zeros = np.zeros((160,160,3), dtype=np.uint8)
    begin_time = time()
    for index in indices:
        score = scores[index]
        class_id = ids[index]
        x1, y1, x2, y2 = dbboxes[index]
        x1_corp, y1_corp, x2_corp, y2_corp = int(x1*x_scale_corp), int(y1*y_scale_corp), int(x2*x_scale_corp), int(y2*y_scale_corp)
        # mask
        print(masks[index].reshape(-1))
        mask = (np.sum(masks[index, :,:,:]*proto[:,y1_corp:y2_corp,x1_corp:x2_corp], axis=0) > 0.0).astype(np.int32)
        zeros[y1_corp:y2_corp,x1_corp:x2_corp, :][mask == 1] = yolo_colors[class_id%20]
        # bbox
        x1, y1, x2, y2 = int(x1*x_scale), int(y1*y_scale), int(x2*x_scale), int(y2*y_scale)
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
    print("\033[0;31;40m" + "Draw Result time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

    begin_time = time()
    zeros = cv2.resize(zeros, (img_w, img_h),cv2.INTER_LANCZOS4)
    result = np.clip(img + 0.5*zeros, 0, 255).astype(np.uint8)
    print("\033[0;31;40m" + "Add Mask time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

    # 保存图片到本地
    begin_time = time()
    if img_h > img_w: # 横着
        cv2.imwrite(opt.test_img + ".result.png", np.hstack((img, zeros, result)))
    else: # 竖着
        cv2.imwrite(opt.test_img + ".result.png", np.vstack((img, zeros, result)))
    print("\033[0;31;40m" + "cv2.imwrite time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

class YOLOv8_Seg_img():
    def __init__(self, opt):
        self.img_path = opt.test_img
        self.result_save_path = opt.test_img + ".result.png"
        self.quantize_model_path = opt.model_path
        self.input_image_size = opt.input_size
        self.classes_num = opt.classes_num
        self.conf=opt.conf_thres
        self.iou=opt.iou_thres
        self.conf_inverse = -np.log(1/self.conf - 1)
        print("iou threshol = %.2f, conf threshol = %.2f"%(self.iou, self.conf))
        print("sigmoid_inverse threshol = %.2f"%self.conf_inverse)
        self.nm = opt.mask_num

        self.sess = HB_ONNXRuntime(model_file=opt.model_path)
                
        # DFL求期望的系数, 只需要生成一次
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, :, np.newaxis]

        # 提前准备一些索引, 只需要生成一次
        self.static_index = np.arange(8400)
        self.s_static_index = np.arange(6400)
        self.m_static_index = np.arange(1600)
        self.l_static_index = np.arange(400)

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0)

    def forward(self, input_img):
        input_names = [input.name for input in self.sess.get_inputs()]
        output_names = [output.name for output in self.sess.get_outputs()]
        feed_dict = dict()
        for input_name in input_names:
            feed_dict[input_name] = self.preprocess(input_img, is_processed=True)
        outputs = self.sess.run(output_names, feed_dict, input_offset=128)
        return self.postprocess(outputs)

    def infer_transformers(self, input_shape, input_layout):
        transformers = [
            # Resize
            ResizeTransformer(target_size=input_shape, mode="opencv"),
            # # BGR->NV12
            # BGR2NV12Transformer(data_format=input_layout[1:], cvt_mode='opencv'),
            # # NV12->YUV444
            # NV12ToYUV444Transformer(target_size=input_shape, yuv444_output_layout=input_layout[1:]),
        ]
        return transformers


    def preprocess(self, image_name, is_processed):
        input_shape = (640, 640)
        input_layout = "NHWC"
        
        transformers = self.infer_transformers(input_shape, input_layout)
        origin_image, processed_image = SingleImageDataLoaderWithOrigin(transformers, image_name, imread_mode="opencv")
        
        processed_image = np.transpose(processed_image, (0, 3, 1, 2))

        processed_image = processed_image.astype(np.uint8)
        return processed_image if is_processed else origin_image
     
    def postprocess(self, quantize_outputs):
        print('the shape of output: ', len(quantize_outputs))
        s_mask_scale = 1
        m_mask_scale = 1
        l_mask_scale = 1
        s_bboxes_scale = 1
        m_bboxes_scale = 1
        l_bboxes_scale = 1
        s_clses_scale = 1
        m_clses_scale = 1
        l_clses_scale = 1
        proto_scale =1

        # 转为numpy
        begin_time = time()
        s_clses = quantize_outputs[0]
        m_clses = quantize_outputs[3]
        l_clses = quantize_outputs[6]
        s_bboxes = quantize_outputs[1]
        m_bboxes = quantize_outputs[4]
        l_bboxes = quantize_outputs[7]
        s_mask = quantize_outputs[2]
        m_mask = quantize_outputs[5]
        l_mask = quantize_outputs[8]
        
        proto = quantize_outputs[9].transpose((0, 3, 1, 2))

        # classify 分支反量化
        s_clses = s_clses.astype(np.float32) * s_clses_scale
        m_clses = m_clses.astype(np.float32) * m_clses_scale
        l_clses = l_clses.astype(np.float32) * l_clses_scale

        # proto 分支的反量化
        proto = (proto*proto_scale)[0]

        # reshape
        s_clses = s_clses[0].reshape(-1, 80).T
        m_clses = m_clses[0].reshape(-1, 80).T
        l_clses = l_clses[0].reshape(-1, 80).T
        s_bboxes = s_bboxes[0].reshape(-1, 64).T
        m_bboxes = m_bboxes[0].reshape(-1, 64).T
        l_bboxes = l_bboxes[0].reshape(-1, 64).T
        s_mask = s_mask[0].reshape(-1, self.nm).T
        m_mask = m_mask[0].reshape(-1, self.nm).T
        l_mask = l_mask[0].reshape(-1, self.nm).T

        # 利用numpy向量化操作完成阈值筛选（优化版）
        s_class_ids = np.argmax(s_clses, axis=0)  # 针对6400行，挑选出80个分数中的最大值的索引
        s_max_scores = s_clses[s_class_ids, self.s_static_index] # 使用最大值的索引索引相应的最大值
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

        m_class_ids = np.argmax(m_clses, axis=0)  # 针对1600行，挑选出80个分数中的最大值的索引
        m_max_scores = m_clses[m_class_ids, self.m_static_index] # 使用最大值的索引索引相应的最大值
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

        l_class_ids = np.argmax(l_clses, axis=0)  # 针对400行，挑选出80个分数中的最大值的索引
        l_max_scores = l_clses[l_class_ids, self.l_static_index] # 使用最大值的索引索引相应的最大值
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

        # 利用筛选结果，索引分数值和id值
        s_scores = s_max_scores[s_valid_indices]
        s_ids = s_class_ids[s_valid_indices]

        m_scores = m_max_scores[m_valid_indices]
        m_ids = m_class_ids[m_valid_indices]

        l_scores = l_max_scores[l_valid_indices]
        l_ids = l_class_ids[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 三个Mask分支的反量化
        s_mask = (s_mask[:,s_valid_indices].astype(np.float32) * s_mask_scale).transpose(1,0)
        m_mask = (m_mask[:,m_valid_indices].astype(np.float32) * m_mask_scale).transpose(1,0)
        l_mask = (l_mask[:,l_valid_indices].astype(np.float32) * l_mask_scale).transpose(1,0)

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[:,s_valid_indices].astype(np.float32) * s_bboxes_scale
        m_bboxes_float32 = m_bboxes[:,m_valid_indices].astype(np.float32) * m_bboxes_scale
        l_bboxes_float32 = l_bboxes[:,l_valid_indices].astype(np.float32) * l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox(ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        s_anchor_indices = self.s_anchor[:,s_valid_indices]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[2:4]
        s_dbboxes = np.vstack([s_x1y1, s_x2y2]).transpose(1,0)*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        m_anchor_indices = self.m_anchor[:,m_valid_indices]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[2:4]
        m_dbboxes = np.vstack([m_x1y1, m_x2y2]).transpose(1,0)*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        l_anchor_indices = self.l_anchor[:,l_valid_indices]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[2:4]
        l_dbboxes = np.vstack([l_x1y1, l_x2y2]).transpose(1,0)*32

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        masks = np.concatenate((s_mask, m_mask, l_mask), axis=0)[:, :, np.newaxis, np.newaxis]

        # nms
        # indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)
        indices = nms(dbboxes, scores, self.iou)

        print('the shape of box: ', dbboxes.shape)
        print('the shape of scores: ', scores.shape)
        print('the shape of masks: ', masks.shape)
        print('the shape of ids: ', ids.shape)
        print('the shape of proto: ', proto.shape)
        print('the shape of indices: ', len(indices))
        
        print(indices)
        return dbboxes, scores, ids, masks, proto, indices

    

  
# 一些常量或函数
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", 
    "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", 
    "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", 
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

yolo_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, box, score, class_id):
    x1, y1, x2, y2 = box
    color = yolo_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )

    # Draw the label text on the image
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()