# Data
This directory contains exemplary (monochromed) images of the employed datasets [Sim](./sim), [KITTI](./kitti) and [Udacity](./udacity) to test our source code. 
The [Udacity](./udacity) sub-directory further contains some test results for object detection, blur estimation and noise estimation.
There is also a [kernels](./kernels) sub-directory with the defocus blur and motion blur kernels used in our experiments. The kernels are given as images and as sampled MTFs.

## Citations

KITTI:
```bibtex
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The {KITTI} Vision Benchmark Suite},
  booktitle = cvpr,
  year = {2012}
}
```

Udacity:
```bibtex
@misc{udacity,
  author = {Udacity},
  title = { },
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/udacity/self-driving-car}},
  commit = {6c3b225}
}
```
