# Neural Network Compression Papers

This is a collection of papers aiming at reducing model sizes or the ASIC/FPGA accelerator for Machine Learning, especially deep neural network related applications.

## Contents

* [Research Background](# research-background)
* [Methods](# methods)
  * [Model Design](# model-design)
  * [Weight Pruning](# weight-pruning)
    * [Unstructured](# unstructured)
    * [Structure](# structured)
  * [Low-rank Factorization](# low-rank-factorization)
  * [Parameter Sharing](# parameter-sharing)
  * [Quantization](# quantization)
  * [Knowledge Distillation](# knowledge-distillation)
  * [Accelerated Computing](# accelerated-computing)

## Research Background

* No significant impact on model prediction accuracy.
* Reduce space complexity by compressing the number of parameters and the depth of the model.
  * There are many parameters in the full connection layer, and the model size is dominated by the full connection layer.
* Does not significantly improve training time while reduce inference time.
  * The convolution layer is computationally intensive and the computational cost is dominated by convolution operations.

## Methods

### Model Design

More refined network structure design

* <u>Aggregated Residual Transformations for Deep Neural Networks</u> [[Paper]](https://arxiv.org/abs/1611.05431) -- We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology
* <u>MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</u> [[Paper]](https://arxiv.org/abs/1704.04861) -- We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks
* <u>MobileNetV2: Inverted Residuals and Linear Bottlenecks</u> [[Paper]](https://arxiv.org/abs/1801.04381) -- The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.
* <u>ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices</u> [[Paper]](https://arxiv.org/abs/1707.01083) -- We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy
* <u>ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design</u> [[Paper]](https://arxiv.org/abs/1807.11164)  -- a new architecture is presented, called \emph{ShuffleNet V2}. Comprehensive ablation experiments verify that our model is the state-of-the-art in terms of speed and accuracy tradeoff.
* <u>Densely Connected Convolutional Networks</u> [[Paper]](https://arxiv.org/abs/1608.06993) -- we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion
* <u>CondenseNet: An Efficient DenseNet using Learned Group Convolutions</u> [[Paper]](https://arxiv.org/abs/1711.09224) -- In this paper we develop CondenseNet, a novel network architecture with unprecedented efficiency. It combines dense connectivity with a novel module called learned group convolution.
* <u>SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size</u> [[Paper]](https://arxiv.org/abs/1602.07360) -- we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet). 
* <u>Highway Networks</u> [[Paper]](https://arxiv.org/abs/1505.00387) --  Highway networks with hundreds of layers can be trained directly using stochastic gradient descent and with a variety of activation functions, opening up the possibility of studying extremely deep and efficient architectures.

### Weight Pruning

Pick out the parameters that are not important in the model and remove them without affecting the effect of the model. Restoring model performance through a retrain process after removing unimportant parameters

How to find an effective evaluation method for the importance of parameters is particularly important in this method. We can also see that this evaluation standard is varied and varied, and it is difficult to judge which method is better.

At present, the method based on weight pruning is the most simple and effective model compression method.

#### Unstructured

* <u>Optimal brain damage</u> [[Paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf) -- By removing unimportant weights from a network, several improvements can be expected: better generalization, fewer training examples required, and improved speed of learning and / or classification.
* <u>Second order derivatives for network pruning: Optimal Brain Surgeon</u> [[Paper]](https://authors.library.caltech.edu/54983/3/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon(1).pdf) -- We investigate the use of information from all second order derivatives of the error function to perfonn network pruning (i.e., removing unimportant weights from a trained network) in order to improve generalization, simplify networks, reduce hardware or storage requirements, increase the speed of further training, and in some cases enable rule extraction. 
* <u>Data-free parameter pruning for Deep Neural Networks</u> [[Paper]](https://arxiv.org/abs/1507.06149) -- In this work, we address the problem of pruning parameters in a trained NN model. Instead of removing individual weights one at a time as done in previous works, we remove one neuron at a time
* <u>Learning both Weights and Connections for Efficient Neural Networks</u> [[Paper]](https://arxiv.org/abs/1506.02626) -- we describe a method to reduce the storage and computation required by neural networks by an order of magnitude without affecting their accuracy by learning only the important connections
* <u>Efficient memory compression in deep neural networks using coarse-grain sparsification for speech applications</u> [[Paper]](https://ieeexplore.ieee.org/document/7827655) --  In this paper, we propose a hardware-centric methodology to design low power neural networks with significantly smaller memory footprint and computation resource requirements. We achieve this by judiciously dropping connections in large blocks of weights.
* <u>Pruning Convolutional Neural Networks for Resource Efficient Inference</u> [[Paper]](https://arxiv.org/abs/1611.06440) -- We propose a new formulation for pruning convolutional kernels in neural networks to enable efficient inference. We interleave greedy criteria-based pruning with fine-tuning by backpropagation - a computationally efficient procedure that maintains good generalization in the pruned network.
* <u>Training Compressed Fully-Connected Networks with a Density-Diversity Penalty</u> [[Paper]](https://openreview.net/pdf?id=Hku9NK5lx) --  In this work, we propose a new “density-diversity penalty” regularizer that can be applied to fully connected layers of neural networks during training

#### Structured

* <u>Learning Structured Sparsity in Deep Neural Networks</u> [[Paper]](https://arxiv.org/abs/1608.03665) --  In this work, we propose a Structured Sparsity Learning (SSL) method to regularize the structures (i.e., filters, channels, filter shapes, and layer depth) of DNNs. SSL can: (1) learn a compact structure from a bigger DNN to reduce computation cost; (2) obtain a hardware-friendly structured sparsity of DNN to efficiently accelerate the DNNs evaluation. Experimental results show that SSL achieves on average 5.1x and 3.1x speedups of convolutional layer computation of AlexNet against CPU and GPU, respectively, with off-the-shelf libraries. These speedups are about twice speedups of non-structured sparsity; (3) regularize the DNN structure to improve classification accuracy. 
* <u>Dynamic Network Surgery for Efficient DNNs</u> [[Paper]](https://arxiv.org/abs/1608.04493) -- In this paper, we propose a novel network compression method called dynamic network surgery, which can remarkably reduce the network complexity by making on-the-fly connection pruning. Unlike the previous methods which accomplish this task in a greedy way, we properly incorporate connection splicing into the whole process to avoid incorrect pruning and make it as a continual network maintenance.
* <u>Structured Pruning of Deep Convolutional Neural Networks</u> [[Paper]](https://arxiv.org/abs/1512.08571) -- We introduce structured sparsity at various scales for convolutional neural networks, which are channel wise, kernel wise and intra kernel strided sparsity. This structured sparsity is very advantageous for direct computational resource savings on embedded computers, parallel computing environments and hardware based systems.
* <u>Channel-level acceleration of deep face representations</u> [[Paper]](https://ieeexplore.ieee.org/document/7303876) -- We propose two novel methods for compression: one based on eliminating lowly active channels and the other on coupling pruning with repeated use of already computed elements. Pruning of entire channels is an appealing idea, since it leads to direct saving in run time in almost every reasonable architecture.
* <u>Pruning Filters for Efficient ConvNets</u> [[Paper]](https://arxiv.org/abs/1608.08710) -- We present an acceleration method for CNNs, where we prune filters from CNNs that are identified as having a small effect on the output accuracy. By removing whole filters in the network together with their connecting feature maps, the computation costs are reduced significantly. In contrast to pruning weights, this approach does not result in sparse connectivity patterns
* <u>Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures</u> [[Paper]](https://arxiv.org/abs/1607.03250) -- These zero activation neurons are redundant, and can be removed without affecting the overall accuracy of the network. After pruning the zero activation neurons, we retrain the network using the weights before pruning as initialization.
* <u>An Entropy-based Pruning Method for CNN Compression</u> [[Paper]](https://arxiv.org/abs/1706.05791) -- This paper aims to simultaneously accelerate and compress off-the-shelf CNN models via filter pruning strategy. The importance of each filter is evaluated by the proposed entropy-based method first. Then several unimportant filters are discarded to get a smaller CNN model. Finally, fine-tuning is adopted to recover its generalization ability which is damaged during filter pruning.
* <u>Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning</u> [[Paper]](https://arxiv.org/abs/1611.05128) -- we propose an energy-aware pruning algorithm for CNNs that directly uses energy consumption estimation of a CNN to guide the pruning process. The energy estimation methodology uses parameters extrapolated from actual hardware measurements that target realistic battery-powered system setups. The proposed layer-by-layer pruning algorithm also prunes more aggressively than previously proposed pruning methods by minimizing the error in output feature maps instead of filter weights.
* <u>Coarse Pruning of Convolutional Neural Networks with Random Masks</u> [[Paper]](https://openreview.net/pdf?id=HkvS3Mqxe) -- we propose a
  simple strategy to choose the least adversarial pruning masks. The proposed approach is generic and can select good pruning masks for feature map, kernel and intra-kernel pruning
* <u>Efficient Gender Classification Using a Deep LDA-Pruned Net</u> [[Paper]](https://arxiv.org/abs/1704.06305) -- Through Fisher's Linear Discriminant Analysis (LDA), we show that this high decorrelation makes it safe to discard directly last conv layer neurons with high within-class variance and low between-class variance. Combined with either Support Vector Machines (SVM) or Bayesian classification, the reduced CNNs are capable of achieving comparable (or even higher) accuracies on the LFW and CelebA datasets than the original net with fully connected layers
* <u>Sparsifying Neural Network Connections for Face Recognition</u> [[Paper]] -- This paper proposes to learn high-performance deep ConvNets with sparse neural connections, referred to as sparse ConvNets
* <u>ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression</u> [[Paper]](https://arxiv.org/abs/1707.06342) -- We propose an efficient and unified framework, namely ThiNet, to simultaneously accelerate and compress CNN models in both training and inference stages. We focus on the filter level pruning, i.e., the whole filter would be discarded if it is less important. Our method does not change the original network structure, thus it can be perfectly supported by any off-the-shelf deep learning libraries.
* <u>Channel Pruning for Accelerating Very Deep Neural Networks</u> [[Paper]](https://arxiv.org/abs/1707.06168) -- In this paper, we introduce a new channel pruning method to accelerate very deep convolutional neural networks.Given a trained CNN model, we propose an iterative two-step algorithm to effectively prune each layer, by a LASSO regression based channel selection and least square reconstruction. 
* <u>Neuron Pruning for Compressing Deep Networks using Maxout Architectures</u> [[Paper]](https://arxiv.org/abs/1707.06838) -- This paper presents an efficient and robust approach for reducing the size of deep neural networks by pruning entire neurons. It exploits maxout units for combining neurons into more complex convex functions and it makes use of a local relevance measurement that ranks neurons according to their activation on the training set for pruning them.
* <u>DeepRebirth: Accelerating Deep Neural Network Execution on Mobile Devices</u> [[Paper]](https://arxiv.org/abs/1708.04728) -- This paper first discovers that the major obstacle is the excessive execution time of non-tensor layers such as pooling and normalization without tensor-like trainable parameters. This motivates us to design a novel acceleration framework: DeepRebirth through "slimming" existing consecutive and parallel non-tensor and tensor layers. 

### Low-rank Factorization

Spread the three-dimensional tensor into a two-dimensional tensor using the standard SVD or other decomposition methods, reducing time complexity of the model

<u>Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition</u> [[Paper]](https://arxiv.org/abs/1412.6553) -- We propose a simple two-step approach for speeding up convolution layers within large convolutional neural networks based on tensor decomposition and discriminative fine-tuning. Given a layer, we use non-linear least squares to compute a low-rank CP-decomposition of the 4D convolution kernel tensor into a sum of a small number of rank-one tensors. 

<u>Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation</u> [[Paper]](https://arxiv.org/abs/1404.0736) -- The computation is dominated by the convolution operations in the lower layers of the model. We exploit the linear structure present within the convolutional filters to derive approximations that significantly reduce the required computation. Using large state-of-the-art models, we demonstrate we demonstrate speedups of convolutional layers on both CPU and GPU by a factor of 2x, while keeping the accuracy within 1% of the original model.

<u>Accelerating Very Deep Convolutional Networks for Classification and Detection</u> [[Paper]](https://arxiv.org/pdf/1505.06798.pdf) -- . We develop an effective solution to the resulting nonlinear optimization problem without the need of stochastic gradient descent (SGD). More importantly, while previous methods mainly focus on optimizing one or two layers, our nonlinear method enables an asymmetric reconstruction that reduces the rapidly accumulated error when multiple (e.g., ≥10) layers are approximated

<u>Speeding up Convolutional Neural Networks with Low Rank Expansions</u> [[Paper]](https://arxiv.org/abs/1405.3866) -- This is achieved by exploiting cross-channel or filter redundancy to construct a low rank basis of filters that are rank-1 in the spatial domain. Our methods are architecture agnostic, and can be easily applied to existing CPU and GPU convolutional frameworks for tuneable speedup performance.

<u>Convolutional neural networks with low-rank regularization</u> [[Paper]](https://arxiv.org/abs/1511.06067?context=cs) -- We propose a new algorithm for computing the low-rank tensor decomposition for removing the redundancy in the convolution kernels. The algorithm finds the exact global optimizer of the decomposition and is more effective than iterative methods. Based on the decomposition, we further propose a new method for training low-rank constrained CNNs from scratch

<u>Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications</u> [[Paper]](https://arxiv.org/abs/1511.06530) -- The proposed scheme consists of three steps: (1) rank selection with variational Bayesian matrix factorization, (2) Tucker decomposition on kernel tensor, and (3) fine-tuning to recover accumulated loss of accuracy, and each step can be easily implemented using publicly available tools.

<u>Fixed-point Factorized Networks</u> [[Paper]](https://arxiv.org/abs/1611.01972) -- we introduce a novel Fixed-point Factorized Networks (FFN) for pretrained models to reduce the computational complexity as well as the storage requirement of networks

<u>Factorized Convolutional Neural Networks</u> [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w10/Wang_Factorized_Convolutional_Neural_ICCV_2017_paper.pdf) -- The 3D convolution operation in a convolutional layer can be considered as performing spatial convolution in each channel and linear projection across channels simultaneously.

### Parameter Sharing

<u>Structured Convolution Matrices for Energy-efficient Deep learning</u> [[Paper]](https://arxiv.org/abs/1606.02407) --  we develop deep convolutional networks using a family of structured convolutional matrices and achieve state-of-the-art trade-off between energy efficiency and classification accuracy for well-known image recognition tasks

<u>Functional Hashing for Compressing Neural Networks</u> [[Paper]](https://arxiv.org/abs/1605.06560) -- This paper presents a novel structure based on functional hashing to compress DNNs, namely FunHashNN. For each entry in a deep net, FunHashNN uses multiple low-cost hash functions to fetch values in the compression space, and then employs a small reconstruction network to recover that entry.

<u>Compressing Deep Convolutional Networks using Vector Quantization</u> [[Paper]](https://arxiv.org/abs/1412.6115?context=cs) -- we tackle this model storage issue by investigating information theoretical vector quantization methods for compressing the parameters of CNNs. In particular, we have found in terms of compressing the most storage demanding dense connected layers, vector quantization methods have a clear gain over existing matrix factorization methods. Simply applying k-means clustering to the weights or conducting product quantization can lead to a very good balance between model size and recognition accuracy.

<u>Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding</u> [[Paper]](https://arxiv.org/abs/1510.00149) -- Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding

### Quantization

Compress the original network by reducing the number of bits required to represent each weight

<u>8-Bit Approximations for Parallelism in Deep Learning</u> [[Paper]](https://arxiv.org/abs/1511.04561) -- we develop and test 8-bit approximation algorithms which make better use of the available bandwidth by compressing 32-bit gradients and nonlinear activations to 8-bit approximations

<u>Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1</u> [[Paper]](https://arxiv.org/abs/1602.02830) -- We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency.

<u>XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks</u> [[Paper]](https://arxiv.org/abs/1603.05279) -- In Binary-Weight-Networks, the filters are approximated with binary values resulting in 32x memory saving. In XNOR-Networks, both the filters and the input to convolutional layers are binary. XNOR-Networks approximate convolutions using primarily binary operations.

<u>Performance Guaranteed Network Acceleration via High-Order Residual Quantization</u> [[Paper]](https://arxiv.org/abs/1708.08687) -- In this paper, we propose a highorder binarization scheme, which achieves more accurate approximation while still possesses the advantage of binary operation. In particular, the proposed scheme recursively performs residual quantization and yields a series of binary input images with decreasing magnitude scales.

<u>Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights</u> [[Paper]](https://arxiv.org/abs/1702.03044) -- The weights in the first group are responsible to form a low-precision base, thus they are quantized by a variable-length encoding method. The weights in the other group are responsible to compensate for the accuracy loss from the quantization, thus they are the ones to be re-trained.

<u>Ternary Weight Networks</u> [[Paper]](https://arxiv.org/abs/1605.04711) -- We introduce ternary weight networks (TWNs) - neural networks with weights constrained to +1, 0 and -1. The Euclidian distance between full (float or double) precision weights and the ternary weights along with a scaling factor is minimized. 

<u>Quantized Convolutional Neural Networks for Mobile Devices</u> [[Paper]](https://arxiv.org/abs/1512.06473) --  we propose an efficient framework, namely Quantized CNN, to simultaneously speed-up the computation and reduce the storage and memory overhead of CNN models. Both filter kernels in convolutional layers and weighting matrices in fully-connected layers are quantized, aiming at minimizing the estimation error of each layer's response.

### Knowledge Distillation 

Transfer learning is the migration of the performance of one model to another, and Knowledge Distillation  is a special case of transfer learning on the same domain.

<u>Distilling the Knowledge in a Neural Network</u> [[Paper]](https://arxiv.org/abs/1503.02531) -- Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. 

<u>Do Deep Nets Really Need to be Deep?</u> [[Paper]](https://arxiv.org/abs/1312.6184) --  In this extended abstract, we show that shallow feed-forward networks can learn the complex functions previously learned by deep nets and achieve accuracies previously only achievable with deep models.

<u>Net2Net: Accelerating Learning via Knowledge Transfer</u> [[Paper]](https://arxiv.org/abs/1511.05641) -- We introduce techniques for rapidly transferring the information stored in one neural net into another neural net. The main purpose is to accelerate the training of a significantly larger neural net

<u>Deep Model Compression: Distilling Knowledge from Noisy Teachers</u> [[Paper]](https://arxiv.org/abs/1610.09650) -- we extend the teacher-student framework for deep model compression, since it has the potential to address runtime and train time complexity too. We propose a simple methodology to include a noise-based regularizer while training the student from the teacher, which provides a healthy improvement in the performance of the student network. 

<u>A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning</u> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) -- we define the distilled knowledge to be transferred in terms of flow between layers, which is calculated by computing the inner product between features from two layers.

<u>Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy</u> [[Paper]](https://arxiv.org/abs/1711.05852) -- In this paper, we study the combination of these two techniques and show that the performance of low-precision networks can be significantly improved by using knowledge distillation techniques. Our approach, Apprentice, achieves state-of-the-art accuracies using ternary precision and 4-bit precision for variants of ResNet architecture on ImageNet dataset

### Accelerated Computing

<u>Faster CNNs with Direct Sparse Convolutions and Guided Pruning</u> [[Paper]](https://blog.csdn.net/qiu931110/article/details/80189905) -- We present a method to realize simultaneously size economy and speed improvement while pruning CNNs. Paramount to our success is an efficient general sparse-with-dense matrix multiplication implementation that is applicable to convolution of feature maps with kernels of arbitrary sparsity patterns. 

<u>MEC: Memory-efficient Convolution for Deep Neural Network</u> [[Paper]](https://arxiv.org/abs/1706.06873) --  we propose a memory-efficient convolution or MEC with compact lowering, which reduces memory-overhead substantially and accelerates convolution process. MEC lowers the input matrix in a simple yet efficient/compact way (i.e., much less memory-overhead), and then executes multiple small matrix multiplications in parallel to get convolution completed. 

