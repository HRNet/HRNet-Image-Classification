# High-resolution networks (HRNets) for Image classification

## Introduction
This is the official code of [high-resolution representations for ImageNet classification](https://arxiv.org/abs/1904.04514). 
We augment the HRNet with a classification head shown in the figure below. First, the four-resolution feature maps are fed into a bottleneck and the number of output channels are increased to 128, 256, 512, and 1024, respectively. Then, we downsample the high-resolution representations by a 2-strided 3x3 convolution outputting 256 channels and add them to the representations of the second-high-resolution representations. This process is repeated two times to get 1024 channels over the small resolution. Last, we transform 1024 channels to 2048 channels through a 1x1 convolution, followed by a global average pooling operation. The output 2048-dimensional representation is fed into the classifier.

![](figures/cls-hrnet.png)

## ImageNet pretrained models
HRNetV2 ImageNet pretrained models are now available!

| model |#Params | GFLOPs |top-1 error| top-5 error| Link |
| :--: | :--: | :--: | :--: | :--: | :--: |
| HRNet-W18-C | 21.3M | 3.99 | 23.2% | 6.6% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw)/[BaiduYun(Access Code:jm34)]( https://pan.baidu.com/s/1B88oBeb-fOuDsAbirFUKZg)|
| HRNet-W30-C | 37.7M | 7.55 | 21.8% | 5.8% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33cQoACCEfrzcSaVI)|
| HRNet-W32-C | 41.2M | 8.31 | 21.5% | 5.8% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w)|
| HRNet-W40-C | 57.6M | 11.8 | 21.1% | 5.5% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo)|
| HRNet-W44-C | 67.1M | 13.9 | 21.1% | 5.6% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33czZQ0woUb980gRs)|
| HRNet-W48-C | 77.5M | 16.1 | 20.7% | 5.5% |[OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk)|

## Quick start
### Install
1. Install PyTorch=0.4.1 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/HRNet/HRNet-Image-Classification
3. Install dependencies: pip install -r requirements.txt

### Data preparation
You can follow the Pytorch implementation:
https://github.com/pytorch/examples/tree/master/imagenet

The data should be under ./data/imagenet/images/.

### Train and test
Please specify the configuration file.

For example, train the HRNet-W18 on ImageNet with a batch size of 128 on 4 GPUs:
````bash
python tools/train.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
````

For example, test the HRNet-W18 on ImageNet on 4 GPUs:
````bash
python tools/valid.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel hrnetv2_w18_imagenet_pretrained.pth
````

## Other applications of HRNet
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) (coming soon)
* [Object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) (coming soon)

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao 
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)
