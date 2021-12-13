# Object Detection

This directory contains the used object detectors YOLOv4 and Faster R-CNN, and the mean Average Precision (mAP) evaluation code.
Please install all requirements first:
```
pip install -r requirements.txt
```

## YOLOv4
Our repository does not contain a trained YOLOv4 model - please download the *yolov4.onnx* file from [YOLOv4 Model](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4) 
and add it under [YOLOv4/model](YOLOv4/model). Following, you may edit and run:
```
python YOLOv4/YOLOv4.py
```

## Faster R-CNN
Same as YOLOv4. Download the Faster R-CNN model file *FasterRCNN-10.onnx* from [Faster R-CNN Model](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn) 
and add it under [fasterRCNN/model](fasterRCNN/model). Then edit and run:
```
python fasterRCNN/fasterRCNN.py
```

## mAP Evaluation
Edit and run:
```
python evaluation/mapEvaluation.py
```

## Citations

YOLOv4:
```bibtex
@article{bochkovskiy2020yolov4,
  title={Yolov4: Optimal speed and accuracy of object detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934}  [Titel anhand dieser ArXiv-ID in Citavi-Projekt übernehmen] ,
  year={2020}
}

@misc{githubMotionBlurKernelCode,
  author = {bddppq et al.},
  title = {ONNX Model Zoo},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/onnx/models}},
  commit = {6ab957a}
}
```

Faster R-CNN:
```bibtex
@article{ren2016faster,
  title={Faster R-CNN: towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={39},
  number={6},
  pages={1137--1149},
  year={2016},
  publisher={IEEE}
}

@misc{githubMotionBlurKernelCode,
  author = {bddppq et al.},
  title = {ONNX Model Zoo},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/onnx/models}},
  commit = {6ab957a}
}
```

mAP Evaluation:
```bibtex
@misc{githubMotionBlurKernelCode,
  author = {Cartucho},
  title = {mAP (mean Average Precision)},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Cartucho/mAP}},
  commit = {3605865}
}
```
