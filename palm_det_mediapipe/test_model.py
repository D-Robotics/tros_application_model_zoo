import copy
import numpy as np
import cv2
from pathlib import Path
from anchor import _load_anchors
from horizon_tc_ui.hb_runtime import HBRuntime

def bgr_to_nv12(bgr_img):
    # 转换为 YUV_I420 (YUV420P)
    yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr_img.shape[:2]

    y_plane = yuv_i420[:h]

    u_plane = yuv_i420[h : h + h // 4].reshape(-1, w // 2)
    v_plane = yuv_i420[h + h // 4 :].reshape(-1, w // 2)

    uv_plane = np.stack((u_plane, v_plane)).transpose(1, 2, 0)
    # uv_plane = uv_plane.reshape(-1, w // 2, 2)

    # uv_plane = np.zeros((h // 2, w // 2, 2), dtype=np.uint8)
    # uv_plane[..., 0] = u_plane  # U channel
    # uv_plane[..., 1] = v_plane  # V channel

    return y_plane, uv_plane

sess = HBRuntime("./model_output_s100/palm_det_192_192_original_float_model.onnx")
# sess = HBRuntime("./model_output_s100/palm_det_192_192_ptq_model.onnx")
# sess = HBRuntime("./model_output_s100/palm_det_192_192_quantized_model.bc")
# sess = HBRuntime("./model_output_s100/palm_det_192_192.hbm")

input_names = sess.input_names
output_names = sess.output_names
suffix = Path(sess.model_path).suffix

ori_image = cv2.imread('./1.png')
original_shape = ori_image.shape

if suffix == '.hbm' or suffix == '.bc':
    image = copy.deepcopy(ori_image)
    image = cv2.resize(image, (192, 192))
    y_plane, uv_plane = bgr_to_nv12(image)
    outputs = sess.run(output_names, {input_names[0]: y_plane[np.newaxis, ..., np.newaxis], input_names[1]: uv_plane[np.newaxis, ...]})
else:
    image = copy.deepcopy(ori_image).astype(np.float32) / 255
    image = cv2.resize(image, (192, 192))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = sess.run(output_names, {input_names[0]: image[np.newaxis, ...]})

anchors = _load_anchors()

score = outputs[1][0, :, 0]
box_delta = outputs[0][0, :, 0:4]
landmark_delta = outputs[0][0, :, 4:]
scale_x = original_shape[1]
scale_y = original_shape[0]
scale = np.array([scale_x, scale_y])

# get scores
score = score.astype(np.float64)
score = 1 / (1 + np.exp(-score))

# get boxes
cxy_delta = box_delta[:, :2] / 192
wh_delta = box_delta[:, 2:] / 192
xy1 = (cxy_delta - wh_delta / 2 + anchors) * scale
xy2 = (cxy_delta + wh_delta / 2 + anchors) * scale
boxes = np.concatenate([xy1, xy2], axis=1)
keep_idx = cv2.dnn.NMSBoxes(boxes, score, 0.5, 0.75, top_k=10)
if len(keep_idx) == 0:
    breakpoint()
score = score[keep_idx]
boxes = boxes[keep_idx]
# boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]

# get landmarks
selected_landmarks = landmark_delta[keep_idx].reshape(-1, 7, 2)
selected_landmarks = selected_landmarks / 192
selected_anchors = anchors[keep_idx]
for idx, landmark in enumerate(selected_landmarks):
    landmark += selected_anchors[idx]
selected_landmarks *= scale
# selected_landmarks -= pad_bias


for i, palm in enumerate(boxes):
    if score[i] > 0.5:
        x1, y1, x2, y2 = palm[0:4]
        cv2.rectangle(ori_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for j, landmark in enumerate(selected_landmarks[i]):
            x, y = landmark
            cv2.circle(ori_image, (int(x), int(y)), 3, (255, 0, 0), -1)
            cv2.putText(ori_image, str(j), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        # cv2.circle(image, (int(palm_box[0]), int(palm_box[1])), (int(palm_box[2]), int(palm_box[3])), (0, 255, 0), 2)
        # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# 保存图片
cv2.imwrite("./output_image.png", ori_image)
