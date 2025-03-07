# hand_gesture_detection

## 编译

### RDK X3

```shell
hbdk-cc --march bernoulli2 -m act_kps_x2_infer-symbol.json -p act_kps_x2_infer-0010.params -s 1x8x21x3 -i ddr --O2 -o gestureDet_8x21.hbm --output-layout NHWC --input-layout NHWC
```

### RDK X5

```shell
hbdk-cc --march bayes-e -m act_kps_x2_infer-symbol.json -p act_kps_x2_infer-0010.params -s 1x8x21x3 -i ddr --O2 -o gestureDet_8x21.hbm --output-layout NHWC --input-layout NHWC
```

### RDK S100

```shell

```