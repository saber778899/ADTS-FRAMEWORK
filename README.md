# ADTS: A Framework for Astronaut Visual Body Parts Detection and Tracking in Space Stations

This is the official implementation of "ADTS: A Framework for Astronaut Visual Body Parts Detection and Tracking in Space Stations" on PyTorch platform.

## Demo

https://github.com/user-attachments/assets/122d6f0a-ac9e-40cf-b90e-9271d4ee0e70

## Illustrations

![image](https://github.com/saber778899/ADTS-FRAMEWORK/blob/main/illustration.png)

## Installation

```python
$ git clone https://github.com/saber778899/ADTS.git
$ pip install -r requirements.txt

# Codes are only evaluated on GTX3090
$ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Datasets

* Download address for the complete dataset and annotations: [AstroVisData](https://drive.google.com/drive/folders/1J6jC7lk71T37W7JEW5QDFIps2e8kAnaL?usp=drive_link),  which includes two subsets: AstroBodyParts and AstroMultiObj.
  
## AstroDetNet Training and Testing

Process official annotations of AstroBodyParts for our task by running 

```python
$ python AstroDetNet/tools/get_anno_HumanParts_v2.py
```

Preparing yolov5-style labels for body-parts

```python
$ python AstroDetNet/utils/labels.py --data data/JointBP_HumanParts.yaml
```

For the training stage, please run:

```python
$ python AstroDetNet/train.py --workers 15 --device 0,1,2,3 --data data/JointBP_HumanParts.yaml \
    --hyp data/hyp-p6.yaml --val-scales 1 --val-flips -1 
```

For the testing stage, please run:

```python
$ python AstroDetNet/demos/image.py
```

## AstroTracNet Training and Testing

Training the RE-ID model with [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ) dataset

```python
$ python AstroTracNet/astro_sort/astro/train.py
```

For the testing stage, please run:

```python
$ python track.py
```

For a quantitative evaluation of AstroTracNet's tracking performance on the AstroMultiObj dataset, please refer to [TrackEval](https://github.com/JonathonLuiten/TrackEval) for details

# Acknowledgement

Our code refers to the following repositories. We thank the authors for releasing the codes.

* [TPAMI 2024 (BPJDetPlus) - BPJDet: Extended Object Representation for Generic Body-Part Joint Detection](https://github.com/hnuzhy/BPJDet/tree/BPJDetPlus?tab=readme-ov-file)

* [TIP 2020 (Hier-RCNN & COCOHumanParts) - Hier R-CNN: Instance-level Human Parts Detection and A New Benchmark](https://github.com/soeaver/Hier-R-CNN)

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)

* [ICIP 2017 - Simple Online and Realtime Tracking with a Deep Association Metric]

* [TMM (IEEE Transactions on Multimedia) - StrongSORT: Make DeepSORT Great Again]([https://github.com/ultralytics/yolov5](https://github.com/dyhBUPT/StrongSORT?tab=readme-ov-file))
