# DO-Conv: Depthwise Over-parameterized Convolutional Layer

Created by Jinming Cao, <a href="http://yangyan.li" target="_blank">Yangyan Li</a>, Mingchao Sun, <a href="https://scholar.google.com/citations?user=NpTmcKEAAAAJ&hl=en" target="_blank">Ying Chen</a>, <a href="https://www.cs.huji.ac.il/~danix/" target="_blank">Dani Lischinski</a>, <a href="https://danielcohenor.com/" target="_blank">Daniel Cohen-Or</a>, <a href="https://cfcs.pku.edu.cn/baoquan/" target="_blank">Baoquan Chen</a>, and <a href="http://irc.cs.sdu.edu.cn/~chtu/index.html" target="_blank">Changhe Tu</a>.

## Introduction

DO-Conv is a depthwise over-parameterized convolutional layer, which can be used as a replacement of conventional convolutional layer in CNNs in the training phase to achieve higher accuracies. In the inference phase, DO-Conv can be fused into a conventional convolutional layer, resulting in the computation amount that is exactly the same as that of a conventional convolutional layer.

Please see our <a href="https://arxiv.org/abs/2006.12030" target="_blank">preprint on arXiv</a> for more details, where we demonstrated the advantages of DO-Conv on various benchmark datasets/tasks.


## DO-Conv Usage

In thie repo, we provide reference implementation of DO-Conv in Tensorflow, PyTorch and GluonCV, as replacement to tf.keras.layers.Conv2D, torch.nn.Conv2d and mxnet.gluon.nn.Conv2D, respectively. Please see the code for more details.
