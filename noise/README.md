
# Noise
This directory contains the source code for [noise estimation](./estimation) and [noise simulation](./simulation).
To install the necessary python requirements run:
```
pip install -r requirements.txt
```

## Estimation
We compared three methods for noise estimation in our experiments: [CNN](./estimation/CNN), [B+F](./estimation/B+F) and [PCA](./estimation/PCA).

### CNN
Customize and run:
```
python estimation/CNN/TFv2/main.py
```
We used the [original training images](https://ece.uwaterloo.ca/~k29ma/exploration/) from the Waterloo dataset to train the CNN.
We suggest to use the upgraded CNN noise estimator for tensorflow version 2 (TFv2).
To recreate the paper's experiments, please use the estimator written for tensorflow version 1 (TFv1).
Please note that the CNN of TFv2 is retrained because we did not manage to transfer the weights, i.e., the TFv1 and TFv2 CNNs do not have the same weights.

### B+F
Customize and run:
```
python estimation/B+F/analyticNoiseEstimation_B+F.py
```

### PCA
Customize and run:
```
python estimation/PCA/analyticNoiseEstimation_PCA.py
```

## Citations
CNN:
```bibtex
@article{tan2019pixelwise,
  title={Pixelwise estimation of signal-dependent image noise using deep residual learning},
  author={Tan, Hanlin and Xiao, Huaxin and Lai, Shiming and Liu, Yu and Zhang, Maojun},
  journal={Computational intelligence and neuroscience},
  volume={2019},
  year={2019},
  publisher={Hindawi}
}

@misc{tan2019pixelwiseGitHub,
  author = {Hanlin Tan},
  title = {Pixel-wise-Estimation-of-Signal-Dependent-Image-Noise},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TomHeaven/Pixel-wise-Estimation-of-Signal-Dependent-Image-Noise-using-Deep-Residual-Learning}},
  commit = {7f2a573}
}
```

B+F:
```bibtex
@article{shin2005block,
  title={Block-based noise estimation using {A}daptive {G}aussian {F}iltering},
  author={Shin, Dong-Hyuk and Park, Rae-Hong and Yang, Seungjoon and Jung, Jae-Han},
  journal={{IEEE} Trans. Consumer Electronics},
  volume={51},
  number={1},
  pages={218--226},
  year={2005}
}
```

PCA:
```bibtex
@InProceedings{Chen15iccv,
  author={Chen, Guangyong and Zhu, Fengyuan and Heng, Pheng Ann},
  booktitle=iccv, 
  title={An Efficient Statistical Method for Image Noise Level Estimation}, 
  year={2015},
  pages={477-485},
  doi={10.1109/ICCV.2015.62}
}

@misc{chen2015efficientGitHub,
  author = {Zongsheng Yue},
  title = {Noise Level Estimation for Signal Image},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zsyOAOA/noise_est_ICCV2015}},
  commit = {a53b4dd}
}
```

Waterloo dataset:
```bibtex
@article{ma2017waterloo,
	author    = {Ma, Kede and Duanmu, Zhengfang and Wu, Qingbo and Wang, Zhou and Yong, Hongwei and Li, Hongliang and Zhang, Lei}, 
	title     = {{Waterloo Exploration Database}: New Challenges for Image Quality Assessment Models}, 
	journal   = {IEEE Transactions on Image Processing},
	volume    = {26},
	number    = {2},
	pages     = {1004--1016},
	month	  = {Feb.},
	year      = {2017}
}
  ```
