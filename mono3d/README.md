# mono3d_indoor_detection

## 编译

### RDK X3

```shell
hbdk-cc --march bernoulli2 -m int-inference-export-last-symbol.json -p int-inference-export-last-0000.params -s 1x512x960x3 -i pyramid --O3 -o centernet.hbm --output-layout NHWC
```

### RDK X5

```shell
hbdk-cc --march bayes-e -m int-inference-export-last-symbol.json -p int-inference-export-last-0000.params -s 1x512x960x3 -i pyramid --O3 -o centernet.hbm --output-layout NHWC
```

### RDK S100

```shell

```