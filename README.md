# Landslide area delimitation via deep transfer learning
#### Mask R-CNN and U-Net are two popular deep learning frameworks for image segmentation in computer vision and have been applied to landslide detection and segmentation in remote sensing images. The objective of this study is to systematically compare and evaluate the performance and adaptability of Mask R-CNN and U-Net for landslide area delimitation in optical remote sensing imagery across regions.

### This code is provided in conjunction with a manuscript submitted to *IEEE Access* in 2024.
#### Title: Delimitation of Landslide Areas in Optical Remote Sensing Images across Regions via Deep Transfer Learning  
#### Authors: Zan Wang, Shengwen Qi, Yu Han, Bowen Zheng, Yu Zou, Yue Yang

#### Python scripts:
 * `landslide_maskrcnn_pretrain_STTCdataset.py` - pre-training the Mask R-CNN model using the [LRSTTC dataset](https://github.com/Jiang-CHD-YunNan/LRSTTC)
 * `landslide_unet_pretrain_STTCdataset.py` - pre-training the U-Net model using the [LRSTTC dataset](https://github.com/Jiang-CHD-YunNan/LRSTTC)
 * `landslide_maskrcnn_transfer.py` - fine-tuning and testing the Mask R-CNN model using the [QTP dataset](https://ieee-dataport.org/documents/qinghai-tibet-plateau-qtp-landslides-dataset)
 * `landslide_unet_transfer.py` - fine-tuning and testing the U-Net model using the [QTP dataset](https://ieee-dataport.org/documents/qinghai-tibet-plateau-qtp-landslides-dataset)
