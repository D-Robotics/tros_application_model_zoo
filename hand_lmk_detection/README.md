# hand_lmk_detection

## 编译

### RDK X3

```shell
hbdk-cc --march bernoulli2 -m model_x2_infer-symbol.json -p model_x2_infer-0199.params -s 1x128x128x3 -i resizer --O2 -o handLMKs.hbm --output-layout NHWC
```

### RDK X5

```shell
hbdk-cc --march bayes-e -m model_x2_infer-symbol.json -p model_x2_infer-0199.params -s 1x128x128x3 -i resizer --O2 -o handLMKs.hbm --output-layout NHWC
```

### RDK S100

```shell

```