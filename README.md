# ADTS: A Framework for Astronaut Visual Body Parts Detection and Tracking in Space Stations

This is the official implementation of "ADTS: A Framework for Astronaut Visual Body Parts \\ Detection and Tracking in Space Stations" on PyTorch platform.

## Demo


Uploading 7月11日_20250711_20022550.mp4…


<table> 
  <tr> 
    <td>
      <img src="https://github.com/saber778899/ADTS-FRAMEWORK/blob/main/demo/Astro1.png">
    </td> 
    <td>
      <img src="https://github.com/saber778899/ADTS-FRAMEWORK/blob/main/demo/Astro2.png">
    </td> 
  </tr> 
</table>

<table> 
  <tr> 
    <td>
      <img src="https://github.com/saber778899/ADTS-FRAMEWORK/blob/main/demo/Astro3.png">
    </td> 
    <td>
      <img src="https://github.com/saber778899/ADTS-FRAMEWORK/blob/main/demo/Astro4.png">
    </td> 
  </tr> 
</table>


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

* Download address for the complete dataset and annotations: [AstroBodyParts Datasets](https://drive.google.com/drive/folders/1wUf7o7ngUXhXjTdUNYcZ4VGuQaP8kzNb?usp=drive_link)

* Download link for the pre-divided training and validation sets: [AstroBodyParts Datasets Pre-divided](https://drive.google.com/drive/folders/1BCIJo1lzVhbxC4TYu6hrPRxHcPhKLde4?usp=drive_link)
  
## Training and Testing

Process official annotations of AstroBodyParts for our task by running 

```python
$ python tools/get_anno_HumanParts_v2.py
```

Preparing yolov5-style labels for body-parts

```python
$ python utils/labels.py --data data/JointBP_HumanParts.yaml
```

For the training stage, please run:

```python
$ python train.py --workers 15 --device 0,1,2,3 --data data/JointBP_HumanParts.yaml \
    --hyp data/hyp-p6.yaml --val-scales 1 --val-flips -1 
```

For the testing stage, please run:

```python
$ python demos/image.py
```


# Acknowledgement

Our code refers to the following repositories. We thank the authors for releasing the codes.

* [TPAMI 2024 (BPJDetPlus) - BPJDet: Extended Object Representation for Generic Body-Part Joint Detection](https://github.com/hnuzhy/BPJDet/tree/BPJDetPlus?tab=readme-ov-file)

* [TIP 2020 (Hier-RCNN & COCOHumanParts) - Hier R-CNN: Instance-level Human Parts Detection and A New Benchmark](https://github.com/soeaver/Hier-R-CNN)

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
