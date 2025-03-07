# elevation

## 编译

### RDK X3

```shell
hbdk-cc --march bernoulli2 -m bpu_model-symbol.json -p bpu_model-0000.params -s 1x512x960x3,1x512x960x3 -o elevation.hbm -i pyramid,pyramid --O3 --output-layout NHWC
```

### RDK X5

```shell
hbdk-cc --march bayes-e -m bpu_model-symbol.json -p bpu_model-0000.params -s 1x512x960x3,1x512x960x3 -o elevation.hbm -i pyramid,pyramid --O3 --output-layout NHWC
```

### RDK S100

```shell

```