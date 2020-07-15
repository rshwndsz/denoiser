# Image Denoising

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ESRhpgv-WdlrXJfKaePBkPJjWszEshQc?usp=sharing)

## References

* Goyal, Bhawna, et al. “Image Denoising Review: From Classical to State-of-the-Art Approaches.” Information Fusion, vol. 55, Mar. 2020, pp. 220–44. DOI.org (Crossref), doi:10.1016/j.inffus.2019.09.003.
* Gu, Shuhang, and Radu Timofte. "A brief review of image denoising algorithms and beyond." Inpainting and Denoising Challenges. Springer, Cham, 2019. 1-21.
* MemNet - https://arxiv.org/pdf/1708.02209.pdf


The model is trained and validated on the images of the [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) dataset.
Testing is then done on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

Validation and Testing results can be seen [here](https://drive.google.com/drive/folders/1p7NcHOwrxXOiTdwwy7KF5Y62nkM7tbSs?usp=sharing)

Validation results are displayed as
```
--------
Denoised 
--------
Noisy
--------
```

Testing results are displayed as
```
Noisy | Denoised | Clean
```
Each section in the testing results has 16 images from CIFAR10.

The trained weights are available [here](https://drive.google.com/file/d/1dLEpBXiAQTBoXHLEgyekoH_FfwN89vQv/view?usp=sharing)

