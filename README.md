# Camera Condition Monitoring and Readjustment by means of Noise and Blur

This repository contains the source code of the paper:
Wischow, M., Gallego, G., Ernst, I., Börner, A., (2021). **[Camera Condition Monitoring and Readjustment by means of Noise and Blur](https://arxiv.org/abs/2112.05456)**.

We propose a modular and general self-health-maintenance framework that strives for optimal application performance. 
We demonstrate the working principle of the framework on the exemplary application of object detection, and focus on motion blur and noise as typical undesired disturbances (see Fig. 1).

![Wischow_fig1](https://user-images.githubusercontent.com/8024432/145776000-333fbc74-f838-4a44-a8c8-62267833dcfd.png)

## General information
- Each sub-directory contains its own readMe file with instructions.
- All python scripts that are intended to be executable have a commented out example code at the end. Before you run any scripts, please uncomment and customize the respective code blocks first. Matlab scripts can be customized and executed directly.
- All python package requirements are specified in the tested versions. Attention: The tensorflow packages of the blur estimation and the noise estimation CNNs are not compatible with each other! Please consider two separate package environments for both.

## Citation
```bibtex
@misc{wischow2021camera,
      title={Camera Condition Monitoring and Readjustment by means of Noise and Blur}, 
      author={Maik Wischow and Guillermo Gallego and Ines Ernst and Anko Börner},
      year={2021},
      eprint={2112.05456},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
