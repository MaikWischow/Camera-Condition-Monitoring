# Blur
This directory contains the source code for [blur estimation](./estimation) and [blur simulation](./simulation).
To install the necessary python requirements run:
```
pip install -r requirements.txt
```

## Estimation
We compared three methods for blur estimation in our experiments: [CNN](./estimation/CNN), [GBA](./estimation/GBA) and [PMP](./estimation/PMP).

### CNN
Customize and run:
```
python estimation/CNN/main.py
```
We used the [original training images](https://ei.is.mpg.de/project/mtf-estimation) to train the CNN.

### GBA
Customize and run:
```
matlab -r estimation/GBA/main -logfile estimation/GBA/logfile
```

### PMP
Customize and run:
```
matlab -r estimation/PMP/main -logfile estimation/PMP/logfile
```

## Citations
CNN:
```
@inproceedings{bauer2028automatic,
  title={Automatic estimation of modulation transfer functions},
  author={Bauer, Matthias and Volchkov, Valentin and Hirsch, Michael and Schc{\"o}lkopf, Bernhard},
  booktitle={IEEE Int. Conf. Comput. Photography (ICCP)},
  year={2018}
}
```

GBA:
```
@article{bai2018graph,
  title={Graph-based blind image deblurring from a single photograph},
  author={Bai, Yuanchao and Cheung, Gene and Liu, Xianming and Gao, Wen},
  journal={IEEE Trans. Image Process.},
  volume={28},
  number={3},
  pages={1404--1418},
  year={2018},
  doi={10.1109/TIP.2018.2874290}
}

@misc{bai2018graphGitHub,
  author = {BYchao100},
  title = {Graph-Based-Blind-Image-Deblurring},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring}},
  commit = {b78dc68}
}
```

PMP:
```
@article{wen2020simple,
  title={A simple local minimal intensity prior and an improved algorithm for blind image deblurring},
  author={Wen, Fei and Ying, Rendong and Liu, Yipeng and Liu, Peilin and Truong, Trieu-Kien},
  journal={IEEE Trans. Circuits Syst. Video Technol.},
  year={2020}
}

@misc{wen2020simpleGitHub,
  author = {FWen},
  title = {deblur-pmp},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FWen/deblur-pmp}},
  commit = {cf6f9e2}
}
```
