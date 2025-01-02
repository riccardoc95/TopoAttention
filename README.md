# TopoAttention

Training module: [TIMM](https://github.com/huggingface/pytorch-image-models)


## Classification
| Dataset                    | Model            | Acc@1                | Acc@5               |
|----------------------------|------------------|----------------------|---------------------|
| MNIST                      | DenseNet         | 98.57                | 99.94               |
|                            | DenseNet with TA | 98.34                | 99.95               |
| CIFAR100                   | DenseNet         | 72.04                | 92.55               |
|                            | DenseNet with TA | 73.93                | 94.10               |
|                            | ResNet18         | 74.36                | 93.05               |
|                            | ResNet18 with TA | 75.77                | 95.31               |
| GALAXY10                   | DenseNet         | 85.96                | 99.46               |
|                            | DenseNet with TA | 87.59                | 99.74               |
|                            | ResNet18         | 86.78                | 99.67               |
|                            | ResNet18 with TA | -                    | -                   |


## Denoising
| Dataset                    | Model            | PSNR                 | SSIM                | MSE                  | NRMSE               |
|----------------------------|------------------|----------------------|---------------------|----------------------|---------------------|
| MNIST ($\sigma = 0.1$)     | UNet             | $29.4053 \pm 1.4213$ | $0.9372 \pm 0.0316$ | $0.0012 \pm 0.0004$  | $0.0343 \pm 0.0057$ |
|                            | UNet with TA     | $30.3535 \pm 1.4663$ | $0.9845 \pm 0.0068$ | $0.0010 \pm 0.0003$  | $0.0308 \pm 0.0050$ |
| MNIST ($\sigma = 0.2$)     | UNet             | $25.9063 \pm 1.4683$ | $0.9608 \pm 0.0152$ | $0.0027 \pm 0.0009$  | $0.0514 \pm 0.0083$ |
|                            | UNet with TA     | $26.0586 \pm 1.4950$ | $0.9597 \pm 0.0151$ | $0.0026 \pm 0.0008$  | $0.0505 \pm 0.0083$ |
| MNIST ($\sigma = 0.5$)     | UNet             | $20.8127 \pm 1.8594$ | $0.8845 \pm 0.0439$ | $0.0098 \pm 0.0034$  | $0.0920 \pm 0.0185$ |
|                            | UNet with TA     | $20.8597 \pm 1.8665$ | $0.8805 \pm 0.0458$ | $0.0089 \pm 0.0035$  | $0.0926 \pm 0.0186$ |
| ILLUSTRIS ($\sigma = 0.1$) | UNet             | $30.6921 \pm 5.7832$ | $0.9110 \pm 0.1038$ | $ 0.0007 \pm 0.0011$ | $0.0458 \pm 0.0513$ |
|                            | UNet with TA     | $34.0265 \pm 4.4771$ | $0.9238 \pm 0.0976$ | $0.0004 \pm 0.0007$  | $0.0268 \pm 0.0231$ |
| ILLUSTRIS ($\sigma = 0.2$) | UNet             | $16.8332 \pm 7.8341$ | $0.8635 \pm 0.1054$ | $ 0.0072 \pm 0.0015$ | $0.2831 \pm 0.3945$ |
|                            | UNet with TA     | $31.4743 \pm 5.2322$ | $0.8187 \pm 0.1763$ | $0.0005 \pm 0.0007$  | $0.0387 \pm 0.0432$ |
| ILLUSTRIS ($\sigma = 0.5$) | UNet             | $24.6280 \pm 6.1283$ | $0.6483 \pm 0.2123$ | $0.0019 \pm 0.0026$  | $0.0945 \pm 0.1051$ |
|                            | UNet with TA     | $28.3992 \pm 5.8176$ | $0.6874 \pm 0.2181$ | $0.0006 \pm 0.0005$  | $0.0583 \pm 0.0623$ |

### MNIST 

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/mnist/report/example01.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/mnist/report/example02.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/mnist/report/example05.png" alt="Alt text" width="500">
</div>

### Galaxy 10 

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example01.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example02.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/galaxy10/report/example05.png" alt="Alt text" width="500">
</div>



### Illustris 

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example01.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example02.png" alt="Alt text" width="500">
</div>


<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/denoising/illustris/report/example05.png" alt="Alt text" width="500">
</div>


## Segmentation
| Dataset                    | Model            | DICE                 | IoU                 | Pixel Accuracy       |
|----------------------------|------------------|----------------------|---------------------|----------------------|
| ILLUSTRIS                  | UNet             | $0.8099 \pm 0.2306$  | $0.6910 \pm 0.2625$ | $0.9755 \pm 0.0225$  |
|                            | UNet with TA     | $0.8386 \pm 0.2249$  | $0.7656 \pm 0.2299$ | $0.9796 \pm 0.0208$  |
| NASA APD Galaxy            | UNet             | $0.6287 \pm 0.2020$  | $0.4867 \pm 0.1947$ | $0.7767 \pm 0.1594$  |
|                            | UNet with TA     | $0.6500 \pm 0.1989$  | $0.5095 \pm 0.1934$ | $0.7806 \pm 0.1547$  |
### MNIST 
| MNIST        | Dice                | IoU                 | Pixel Accuracy      |
|--------------|---------------------|---------------------|---------------------|
| unet         | 0.9986 $\pm$ 0.0029 | 0.9972 $\pm$ 0.0057 | 0.9995 $\pm$ 0.0008 |
| unet_topoatt | 0.9981 $\pm$ 0.0035       | 0.9963 $\pm$ 0.0069       | 0.9994 $\pm$ 0.0009       |

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/segmentation/mnist/report/example.png" alt="Alt text" width="500">
</div>

### Illustris 

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/segmentation/illustris/report/example.png" alt="Alt text" width="500">
</div>

### GalaxySegment

<div align="left">
  <img src="https://github.com/riccardoc95/TopoAttention/blob/main/tasks/segmentation/galaxysegment/report/example.png" alt="Alt text" width="500">
</div>
