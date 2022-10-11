`` python main.py -t -a resnet18 --pretrained C:\Users\Local_Admin\Work\Dataset\dataset ``

```vb
2022-10-11 11:24:29 [INFO] Pass query framework capability elapsed time: 16.09 ms
2022-10-11 11:24:29 [INFO] Get FP32 model baseline.
2022-10-11 11:25:10 [INFO] Save tuning history to C:\Users\Local_Admin\Work\Codes\neural-compressor\examples\pytorch\image_recognition\torchvision_models\quantization\ptq\cpu\eager\nc_workspace\2022-10-11_11-24-27\./history.snapshot.
2022-10-11 11:25:10 [INFO] FP32 baseline is: [Accuracy: 0.6996, Duration (seconds): 40.7074]
2022-10-11 11:25:10 [WARNING] Please note that calibration sampling size 300 isn't divisible exactly by batch size 256. So the real sampling size is 512.
2022-10-11 11:26:05 [INFO] |******Mixed Precision Statistics*****|
2022-10-11 11:26:05 [INFO] +-----------------+----------+--------+
2022-10-11 11:26:05 [INFO] |     Op Type     |  Total   |  INT8  |
2022-10-11 11:26:05 [INFO] +-----------------+----------+--------+
2022-10-11 11:26:05 [INFO] |    ConvReLU2d   |    9     |   9    |
2022-10-11 11:26:05 [INFO] |      Conv2d     |    11    |   11   |
2022-10-11 11:26:05 [INFO] |     add_relu    |    8     |   8    |
2022-10-11 11:26:05 [INFO] |      Linear     |    1     |   1    |
2022-10-11 11:26:05 [INFO] |     Quantize    |    1     |   1    |
2022-10-11 11:26:05 [INFO] |    DeQuantize   |    1     |   1    |
2022-10-11 11:26:05 [INFO] +-----------------+----------+--------+
2022-10-11 11:26:05 [INFO] Pass quantize model elapsed time: 55481.09 ms
2022-10-11 11:26:31 [INFO] Tune 1 result is: [Accuracy (int8|fp32): 0.7026|0.6996, Duration (seconds) (int8|fp32): 26.2519|40.7074], Best tune result is: [Accuracy: 0.7026, Duration (seconds): 26.2519]
2022-10-11 11:26:31 [INFO] |**********************Tune Result Statistics**********************|
2022-10-11 11:26:31 [INFO] +--------------------+----------+---------------+------------------+
2022-10-11 11:26:31 [INFO] |     Info Type      | Baseline | Tune 1 result | Best tune result |
2022-10-11 11:26:31 [INFO] +--------------------+----------+---------------+------------------+
2022-10-11 11:26:31 [INFO] |      Accuracy      | 0.6996   |    0.7026     |     0.7026       |
2022-10-11 11:26:31 [INFO] | Duration (seconds) | 40.7074  |    26.2519    |     26.2519      |
2022-10-11 11:26:31 [INFO] +--------------------+----------+---------------+------------------+
2022-10-11 11:26:31 [INFO] Save tuning history to C:\Users\Local_Admin\Work\Codes\neural-compressor\examples\pytorch\image_recognition\torchvision_models\quantization\ptq\cpu\eager\nc_workspace\2022-10-11_11-24-27\./history.snapshot.
2022-10-11 11:26:31 [INFO] Specified timeout or max trials is reached! Found a quantized model which meet accuracy goal. Exit.
2022-10-11 11:26:31 [INFO] Save deploy yaml to C:\Users\Local_Admin\Work\Codes\neural-compressor\examples\pytorch\image_recognition\torchvision_models\quantization\ptq\cpu\eager\nc_workspace\2022-10-11_11-24-27\deploy.yaml
2022-10-11 11:26:32 [INFO] Save config file and weights of quantized model to C:\Users\Local_Admin\Work\Codes\neural-compressor\examples\pytorch\image_recognition\torchvision_models\quantization\ptq\cpu\eager\saved_results.
```

`` python main.py -e -a resnet18 --pretrained C:\Users\Local_Admin\Work\Dataset\dataset``

```vb
Have createn the dataloader
Test: [0/4]     Time  0.000 ( 0.000)    Loss 1.3372e+00 (1.3372e+00)    Acc@1  67.97 ( 67.97)   Acc@5  87.11 ( 87.11)
```