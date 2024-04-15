# CNVR

## About
[CNVR](https://xplorestaging.ieee.org/document/10443640) is a viewing ray restriction and CNCC guided multi-view stereo method for depth map estimation. If you find this project useful for your research, please cite:  

```
@ARTICLE{10443640,
  title={Multiview Stereo via Noise Suppression PatchMatch}, 
  author={Chen, Jun and Zhu, Aoyu and Yu, Yang and Ma, Jiayi},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  year={2024}
}
```

## Dependencies
The code has been tested on Windows 10 with GTX GeForce 3090.  
* [Cuda](https://developer.nvidia.com/zh-cn/cuda-downloads) >= 6.0
* [OpenCV](https://opencv.org/) >= 2.4
* [cmake](https://cmake.org/)

## Usage
* Compile CNVR

1. Configure and generate project in Cmake GUI
2. run build.bat


* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to CNVR input   
Run ./CNVR $data_folder to get reconstruction results
Run NCD.py to get visualization results of medium results
```

## Results on high-res ETH3D training dataset [2cm]


| Mean   | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes  | playgroud | relief | relief_2 | terrace | terrains |
|--------|-----------|---------------|---------|--------|--------|--------|--------|--------|-----------|--------|----------|---------|----------|
| 86.36  | 91.33     | 89.84         | 90.34   | 75.56  | 87.36  | 78.97  | 81.19	 | 86.76  |  78.74    | 89.30  |  88.58   | 90.45	  | 94.33    |


## Acknowledgements
This code largely benefits from the following repositories: [Gipuma](https://github.com/kysucix/gipuma), [COLMAP](https://colmap.github.io/) and [ACMM](https://github.com/GhiXu/ACMM). Thanks to their authors for opening source of their excellent works.
