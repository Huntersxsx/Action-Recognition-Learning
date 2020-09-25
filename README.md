## <font face="楷体">说明</font>
**动作识别(Action Recognition)任务中常见的模型Pytorch实现**   

## <font face="楷体">主要模型</font>
**3D卷积类**
- **C3D**：[Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf) -*D.Tran et al, ICCV 2015*. 
- **I3D**：[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) -*J.Carreira et al, CVPR 2017*.
- **P3D**：[Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](https://arxiv.org/pdf/1711.10305.pdf) -*Z.Qui et al, ICCV 2017*.
- **R(2+1)D**：[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/pdf/1711.11248.pdf) -*D.Tran et al， CVPR 2018*.
- **3D ResNets**：[Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf) -*K.Hara et al, CVPR 2019*.
 
**Two Stream类**
- **Two Stream**：[Two-Stream Convolutional Networks for Action Recognition in Videos](https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf) -*K.Simonyan and A.Zisserman, NIPS 2014*.
- **Two Stream Fused**：[Convolutional Two-Stream Network Fusion for Video Action Recognition](https://arxiv.org/pdf/1604.06573.pdf) -*C.Feichtenhofer et al, CVPR 2016*.
- **TSN**：[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf) -*L.Wang et al, arXiv 2016*

**CNN+RNN类**
- **LRCN**：[Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/pdf/1411.4389.pdf) -*J.Donahue et al, CVPR 2015*.
- **ConvLSTM**：[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/pdf/1506.04214.pdf) -*X Shi et al, NIPS 2015*.

## <font face="楷体">参考</font>
- R(2+1)D: https://github.com/irhum/R2Plus1D-PyTorch
- 多种模型实现: https://github.com/MRzzm/action-recognition-models-pytorch
- R(2+1)D: https://github.com/kenshohara/3D-ResNets-PyTorch
- C3D: https://github.com/jfzhang95/pytorch-video-recognition
- HOG：https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
- HOF：https://blog.csdn.net/wsp_1138886114/article/details/84400392
