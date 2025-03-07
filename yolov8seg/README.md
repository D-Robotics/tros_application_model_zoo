# yolov8seg

## 编译

### RDK X5

```shell
hb_mapper makertbin --config yolov8n_seg_config.yaml --model-type onnx
```

### RDK S100

```shell
hb_compile --config yolov8n_seg_config_s100.yaml
```