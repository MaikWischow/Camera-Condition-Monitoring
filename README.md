# Monitoring and Adapting the Physical State of a Camera for Autonomous Vehicles (IEEE T-ITS, 2023)

This repository contains the source code of the paper **[Monitoring and Adapting the Physical State of a Camera for Autonomous Vehicles](https://doi.org/10.1109/TITS.2023.3328811) (IEEE T-ITS, 2023)**,  by [Maik Wischow](https://www.researchgate.net/profile/Maik-Wischow), [Guillermo Gallego](https://sites.google.com/view/guillermogallego), [Ines Ernst](https://www.researchgate.net/profile/Ines-Ernst) and [Anko Börner](https://www.researchgate.net/profile/Anko-Boerner). 

A postprint PDF of the paper, including supplementary material, [is available on arXiv](https://arxiv.org/pdf/2112.05456).

We propose a modular and general self-health-maintenance framework that strives for optimal application performance. 
We demonstrate the working principle of the framework on the exemplary application of object detection, and focus on motion blur and noise as typical undesired disturbances (see Fig. 1).

![MaikWischow_Figure1](https://user-images.githubusercontent.com/93527304/207606237-2ed213b7-7aab-4958-9e10-06eb6cd14797.png)

## General information
- Each sub-directory contains its own readMe file with instructions.
- All python scripts that are intended to be executable have a commented out example code at the end. Before you run any scripts, please uncomment and customize the respective code blocks first. Matlab scripts can be customized and executed directly.
- All python package requirements are specified in the tested versions.

## Citation
```bibtex
@article{wischow2023monitoring,
  author = {Wischow, Maik and Gallego, Guillermo and Ernst, Ines and Börner, Anko},
  title = {Monitoring and Adapting the Physical State of a Camera for Autonomous Vehicles},
  journal = {IEEE Transactions on Intelligent Transportation Systems (T-ITS)},
  year = {2023},
  volume = {},
  number = {},
  pages = {1-14},
  doi = {10.1109/TITS.2023.3328811}
}
