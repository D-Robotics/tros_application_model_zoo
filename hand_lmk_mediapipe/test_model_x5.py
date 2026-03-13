import numpy as np
import cv2
import copy
from pathlib import Path
# 加载地平线依赖库
from horizon_tc_ui import HB_ONNXRuntime as HBRuntime

def bgr_to_nv12(bgr_img):
    # 转换为 YUV_I420 (YUV420P)
    yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr_img.shape[:2]

    y_plane = yuv_i420[:h]

    u_plane = yuv_i420[h : h + h // 4].reshape(-1, w // 2)
    v_plane = yuv_i420[h + h // 4 :].reshape(-1, w // 2)

    uv_plane = np.stack((u_plane, v_plane)).transpose(1, 2, 0)

    return y_plane, uv_plane

# sess = HBRuntime("./model_output_x5/hand_224_224_original_float_model.onnx")
sess = HBRuntime("./model_output_x5/hand_224_224_quantized_model.onnx")

input_names = sess.input_names
output_names = sess.output_names

ori_image = cv2.imread('./p5.png')
original_shape = ori_image.shape

image = copy.deepcopy(ori_image)
image = cv2.resize(image, (224, 224))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

output = sess.run(output_names, {input_names[0]: image[np.newaxis, ...]})


# if suffix == '.onnx':
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image.astype(np.float32) / 255
# elif suffix == '.hbm' or suffix == '.bc':
#     scale = 1/255
#     zero_point = -128

#     image = image.astype(np.float32) / 255
    # image = np.round(image / scale + zero_point).astype(np.int8)
    # image = image.astype(np.float32) - 128
    # image = image.astype(np.int8)

# input = np.load('kps/aircanvas/calibration_data_bgr_s100/input_1/IMG_20220430_181047.npy')


keypoints = output[0].reshape(-1, 3)
score = output[1][0][0]

score2 = output[2][0][0]

print(f'{score=},{score2=}')

# 在原始图片上绘制关键点
for i, (x, y, z) in enumerate(keypoints):
    x = int(x * ori_image.shape[1] / 224)
    y = int(y * ori_image.shape[0] / 224)
    cv2.circle(ori_image, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(ori_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# 保存图片
cv2.imwrite("./output_5_.png", ori_image)
