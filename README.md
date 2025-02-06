### Method Overview

The main architecture of our method is illustrated in Figure 1. The input passes through a U-Net backbone to extract multi-scale features, which are then fused using a channel-wise self-attention module. These features are further processed using dynamic snake convolution with tubular offsets. Finally, the segmentation output is generated through a classification head, composed of KAN layers.

![Framework](./figs/SnakeKanFramework.jpg)

### Dynamic Snake Convolution for Bridge Crack Detection

In our bridge crack detection task, we introduce Dynamic Snake Convolution (DSC) [Qi et al., 2023] to enhance boundary delineation and feature extraction. Given an input tensor \(\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}\), where \(B\) is the batch size, \(C\) is the number of channels, and \(H \times W\) represents the spatial dimensions, DSC dynamically adjusts the offsets of convolutional kernels based on the geometric properties of the cracks, as shown in Figure 2.

On the left is the standard convolution, and on the right is the dynamic snake convolution. DSC, with tubular offsets, allows the network to focus more on the intricate details of tubular structures, leading to improved crack detection.

![SnakeKANDSC](./figs/SnakeKANDSC_00.jpg)
![SnakeKANDataset](./figs/SnakeKANDataset_00.jpg)

### Ablation Study Results

Figure 3 presents the visual results of our ablation study, comparing the performance of the method with and without the DSC module. The top row shows crack detection results without the DSC module, where the cracks are less continuous and more fragmented. In contrast, the bottom row shows the results with the DSC module, where cracks are detected with greater continuity and accuracy. These visual results are further supported by the quantitative improvements displayed in Table 1, demonstrating the significant impact of the DSC module in enhancing crack detection performance.

![SnakeKANAblation](./figs/SnakeKANAblation_00.jpg)
![SnakeKANResultsCompare](./figs/SnakeKANResultsCompare_00.jpg)

### Citation

If you use this method in your work, please cite our paper:

Ruan, Y.; Wang, D.; Yuan, Y.; Jiang, S.; Yang, X. SKPNet: Snake KAN Perceive Bridge Cracks Through Semantic Segmentation. *Intell. Robot.* 2025, 5(1), 105-118. [http://dx.doi.org/10.20517/ir.2025.0](http://dx.doi.org/10.20517/ir.2025.0)
