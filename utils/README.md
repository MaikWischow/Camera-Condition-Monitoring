# Utils

This directory contains utility source code. Hence, make sure to install the requirements from the [blur](../blur), [noise](../noise) and [objectDetection](../objectDetection) directories first.

The only standalone script is kernelImg2MTF.py. You may use this file to convert a blur kernel image into the Fourier space and sample the Modulation Transfer Function at the frequencies used in our experiments:
```
python kernelImg2MTF.py
```
