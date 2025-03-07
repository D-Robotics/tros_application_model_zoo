# mono2d_body_detection

## 编译

### RDK X3

```shell
hbdk-cc --march bernoulli2 \
    -m multitask_x2_infer_person_head_face_hand_kps_limit_rcnn_30boxes_v0.1.1_w960xh960.json \
    -p multitask_x2_infer-0010.params \
    -s 1x544x960x3 -i pyramid --O2 -o multitask_body_head_face_hand_kps_960x544.hbm \
     --output-layout NHWC
```

### RDK X5

```shell
hbdk-cc --march bayes-e \
    -m multitask_x2_infer_person_head_face_hand_kps_limit_rcnn_30boxes_v0.1.1_w960xh960.json \
    -p multitask_x2_infer-0010.params \
    -s 1x544x960x3 -i pyramid --O2 -o multitask_body_head_face_hand_kps_960x544.hbm \
     --output-layout NHWC
```

### RDK S100

```shell

```