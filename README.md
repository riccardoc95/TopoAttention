# TopoAttention

Training module: [TIMM](https://github.com/huggingface/pytorch-image-models)


## Classification

| Cifar100         | acc@1 | acc@5 |
|---------------|-------|-------|
| densenet      | 57.36 | 83.08 |
| topo_densenet | 59.00 | 84.77 |

| Galaxy10         | acc@1 | acc@5 |
|------------------|-------|-------|
| densenet         | 85.96 | 99.46 |
| densenet_topoatt | 87.59 | 99.74 |


## Denoising
### Galaxy 10 

| Galaxy10 ($\sigma=0.1$) | PSNR    | SSIM   | MSE    | NRMSE  |
|--------------|---------|--------|--------|--------|
| unet         | 28.4029 | 0.6948 | 0.0014 | 0.0381 |
| unet_topoatt | 28.5438 | 0.6920 | 0.0012 | 0.0372 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example01.png" alt="Alt text" width="500">
</div>


| Galaxy10 ($\sigma=0.2$) | PSNR    | SSIM   | MSE    | NRMSE  |
|-------------------------|---------|--------|--------|--------|
| unet                    | 23.0851 | 0.4755 | 0.0049 | 0.0702 |
| unet_topoatt            | 23.4293 | 0.4685 | 0.0048 | 0.0759 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example02.png" alt="Alt text" width="500">
</div>



| Galaxy10 ($\sigma=0.2$) | PSNR    | SSIM   | MSE    | NRMSE  |
|-------------------------|---------|--------|--------|--------|
| unet                    | 21.8421 | 0.3778 | 0.0066 | 0.0811 |
| unet_topoatt            | 21.5456 | 0.3657 | 0.0071 | 0.0841 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example05.png" alt="Alt text" width="500">
</div>



### Illustris 

| Illustris ($\sigma=0.1$) | PSNR    | SSIM   | MSE    | NRMSE  |
|---------------------------|---------|--------|--------|--------|
| unet                      | 30.6811 | 0.9103 | 0.0007 | 0.0457 |
| unet_topoatt  | 34.0184 | 0.9230 | 0.0004 | 0.0267 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example01.png" alt="Alt text" width="500">
</div>


| Illustris  ($\sigma=0.2$)  | PSNR    | SSIM   | MSE    | NRMSE  |
|----------------------------|---------|--------|--------|--------|
| unet         | 16.8410 | 0.8640 | 0.0073 | 0.2834 |
| unet_topoatt  | 31.4819 | 0.8193 | 0.0005 | 0.0386 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example02.png" alt="Alt text" width="500">
</div>

| Illustris   ($\sigma=0.5$) | PSNR    | SSIM   | MSE    | NRMSE  |
|---------------------------|---------|--------|--------|--------|
| unet         | 24.6351 | 0.6488 | 0.0019 | 0.0944 |
| unet_topoatt | 28.4069 | 0.6878 | 0.0006 | 0.0584 |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example05.png" alt="Alt text" width="500">
</div>


## Segmentation
### Illustris 
| Illustris    | Dice   | IoU    | Pixel Accuracy |
|--------------|--------|--------|----------------|
| unet         | 0.8099 | 0.6910 | 0.9755         |
| unet_topoatt | 0.8386 | 0.7656 | 0.9796         |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/segmentation/illustris/report/example.png" alt="Alt text" width="500">
</div>

### GalaxySegment
| GalaxySegment | Dice   | IoU    | Pixel Accuracy |
|---------------|--------|--------|----------------|
| unet          | 0.6287 | 0.4867 | 0.7767         |
| unet_topoatt  | 0.6500 | 0.5095 | 0.7806         |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/segmentation/galaxysegment/report/example.png" alt="Alt text" width="500">
</div>