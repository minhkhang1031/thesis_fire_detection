# Fire Detection

This project is my thesis in university. My goal is building a device that can detect fire through out camera and model deep learning.

# Model

I train 3 model are: SSD MobileNetV2, SSD EfficientDetD0, and YOLO11n. And final I choose YOLO11n because it have best performace.

I convert the format of model YOLO11n from Pytorch into NCNN to reduce inference time on device.

# Device

I use Raspberry Pi 4 model B 4GB for main platform. Beside, I use camera OV5647 with 2 sensors: MQ-2 (smokes) and DHT22 (temperature)

# Demo


https://github.com/user-attachments/assets/aff23fab-cffc-4b5d-9304-464672137179

