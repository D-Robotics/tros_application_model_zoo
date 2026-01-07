# yolov8n

## 编译

### RDK X5

```shell
hb_mapper makertbin --config yolov8n_config.yaml --model-type onnx
```

### RDK S100

```shell
hb_compile --config yolov8n_config_s100.yaml
```

### RDK S600

```shell
hb_compile --config yolov8n_config_s600.yaml
python3 j6p_int_demo.py
```