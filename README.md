# DO-Conv: Depthwise Over-parameterized Convolutional Layer

Created by Jinming Cao, <a href="http://yangyan.li" target="_blank">Yangyan Li</a>, Mingchao Sun, <a href="https://scholar.google.com/citations?user=NpTmcKEAAAAJ&hl=en" target="_blank">Ying Chen</a>, <a href="https://www.cs.huji.ac.il/~danix/" target="_blank">Dani Lischinski</a>, <a href="https://danielcohenor.com/" target="_blank">Daniel Cohen-Or</a>, <a href="https://cfcs.pku.edu.cn/baoquan/" target="_blank">Baoquan Chen</a>, and <a href="http://irc.cs.sdu.edu.cn/~chtu/index.html" target="_blank">Changhe Tu</a>.

## Introduction

DO-Conv is a depthwise over-parameterized convolutional layer, which can be used as a replacement of conventional convolutional layer in CNNs in the training phase to achieve higher accuracies. In the inference phase, DO-Conv can be fused into a conventional convolutional layer, resulting in the computation amount that is exactly the same as that of a conventional convolutional layer.

Please see our <a href="https://arxiv.org/abs/2006.12030" target="_blank">preprint on arXiv</a> for more details, where we demonstrated the advantages of DO-Conv on various benchmark datasets/tasks.

## We Highly Welcome Issues

**We highly welcome issues, rather than emails, for DO-Conv related questions.**

Moreover, it would be great if a **minimal reproduciable example code** is provide in the issue.

## ImageNet Classification Performance

We take the <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html" target="_blank">model zoo</a> of <a href="https://gluon-cv.mxnet.io/contents.html" target="_blank">GluonCV</a> as baselines. The settings in the baselines have been tuned to favor baselines, and they are not touched during the switch to DO-Conv. In other words, DO-Conv is the one and only change over baselines, and no hyper-parameter tuning is conducted to favor DO-Conv. We consider GluonCV highly reproducible, but still, to exclude clutter factors as much as possible, we train the baselines ourselves, and compare DO-Conv versions with them, while reporting the performance provided by GluonCV as reference. The results are summarized in this table where the “DO-Conv” column shows the performance gain over the baselines.
<table>
<thead>
  <tr>
    <th>Network</th>
    <th>Depth</th>
    <th>Reference</th>
    <th>Baseline</th>
    <th>DO-Conv</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Plain</td>
    <td>18</td>
    <td>-</td>
    <td>69.97</td>
    <td>+1.01</td>
  </tr>
  <tr>
    <td rowspan="5">ResNet-v1</td>
    <td>18</td>
    <td>70.93</td>
    <td>70.87</td>
    <td>+0.82</td>
  </tr>
  <tr>
    <td>34</td>
    <td>74.37</td>
    <td>74.49</td>
    <td>+0.49</td>
  </tr>
  <tr>
    <td>50</td>
    <td>77.36</td>
    <td>77.32</td>
    <td>+0.08</td>
  </tr>
  <tr>
    <td>101</td>
    <td>78.34</td>
    <td>78.16</td>
    <td>+0.46</td>
  </tr>
  <tr>
    <td>152</td>
    <td>79.22</td>
    <td>79.34</td>
    <td>+0.07</td>
  </tr>
  <tr>
    <td rowspan="5">ResNet-v1b</td>
    <td>18</td>
    <td>70.94</td>
    <td>71.08</td>
    <td>+0.71</td>
  </tr>
  <tr>
    <td>34</td>
    <td>74.65</td>
    <td>74.35</td>
    <td>+0.77</td>
  </tr>
  <tr>
    <td>50</td>
    <td>77.67</td>
    <td>77.56</td>
    <td>+0.44</td>
  </tr>
  <tr>
    <td>101</td>
    <td>79.20</td>
    <td>79.14</td>
    <td>+0.25</td>
  </tr>
  <tr>
    <td>152</td>
    <td>79.69</td>
    <td>79.60</td>
    <td>+0.10</td>
  </tr>
  <tr>
    <td rowspan="5">ResNet-v2</td>
    <td>18</td>
    <td>71.00</td>
    <td>70.80</td>
    <td>+0.64</td>
  </tr>
  <tr>
    <td>34</td>
    <td>74.40</td>
    <td>74.76</td>
    <td>+0.22</td>
  </tr>
  <tr>
    <td>50</td>
    <td>77.17</td>
    <td>77.17</td>
    <td>+0.31</td>
  </tr>
  <tr>
    <td>101</td>
    <td>78.53</td>
    <td>78.56</td>
    <td>+0.11</td>
  </tr>
  <tr>
    <td>152</td>
    <td>79.21</td>
    <td>79.24</td>
    <td>+0.14</td>
  </tr>
  <tr>
    <td>ResNext</td>
    <td>50_32x4d</td>
    <td>79.32</td>
    <td>79.21</td>
    <td>+0.40</td>
  </tr>
  <tr>
    <td>MobileNet-v1</td>
    <td>-</td>
    <td>73.28</td>
    <td>73.30</td>
    <td>+0.03</td>
  </tr>
  <tr>
    <td>MobileNet-v2</td>
    <td>-</td>
    <td>72.04</td>
    <td>71.89</td>
    <td>+0.16</td>
  </tr>
  <tr>
    <td>MobileNet-v3</td>
    <td>-</td>
    <td>75.32</td>
    <td>75.16</td>
    <td>+0.14</td>
  </tr>
</tbody>
</table>

## DO-Conv Usage

In thie repo, we provide reference implementation of DO-Conv in <a href="https://www.tensorflow.org/" target="_blank">Tensorflow</a> (tensorflow-gpu==2.2.0), <a href="https://pytorch.org/" target="_blank">PyTorch</a> (pytorch==1.4.0, torchvision==0.5.0) and <a href="https://gluon-cv.mxnet.io/contents.html" target="_blank">GluonCV</a> (mxnet-cu100==1.5.1.post0, gluoncv==0.6.0), as replacement to <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D" target="_blank">tf.keras.layers.Conv2D</a>, <a href="https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html" target="_blank">torch.nn.Conv2d</a> and <a href="https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.Conv2D.html" target="_blank">mxnet.gluon.nn.Conv2D</a>, respectively. Please see the code for more details.

We highly welcome pull requests for adding support for different versions of Pytorch/Tensorflow/GluonCV.

## Example Usage: Tensorflow (tensorflow-gpu==2.2.0)
We show how to use DO-Conv based on the examples provided in the <a href="https://www.tensorflow.org/tutorials/quickstart/advanced" target="_blank">Tutorial</a> of TensorFlow with MNIST dataset.

1 . Run the demo example first to get the accuracy of the baseline.
````
python sample_tf.py
````
If there is any wrong at this step, please check whether the tensorflow version meets the requirements.

2 . Replace these lines:
````
self.conv1 = Conv2D(32, 3, activation='relu')
self.conv2 = Conv2D(8, 3, activation='relu')
````
with
````
self.conv1 = DOConv2D(32, 3, activation='relu')
self.conv2 = DOConv2D(8, 3, activation='relu')
````
to apply DO-Conv without any other changes. 
````
python sample_tf.py
````
3 . We provide the performance improvement in this demo example as follows. (averaged accuracy (%) of five runs)

|          | run1  | run2  | run3  | run4  | run5  | avg    | +     |
|----------|-------|-------|-------|-------|-------|--------|-------|
| Baseline | 98.5  | 98.51 | 98.54 | 98.46 | 98.51 | 98.504 | -     |
| DO-Conv  | 98.71 | 98.62 | 98.67 | 98.75 | 98.66 | 98.682 | 0.178 |

4 . Then you can use DO-Conv in your own network in this way.

## Example Usage: PyTorch (pytorch==1.4.0, torchvision==0.5.0)
We show how to use DO-Conv based on the examples provided in the <a href="https://pytorch.org/tutorials/beginner/nn_tutorial.html?highlight=mnist" target="_blank">Tutorial</a> of PyTorch with MNIST dataset.

1 . Run the demo example first to get the accuracy of the baseline.
````
python sample_pt.py
````
If there is any wrong at this step, please check whether the pytorch and torchvision versions meets the requirements.

2 . Replace these lines:
````
model = nn.Sequential(
    Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
````
with
````
model = nn.Sequential(
    DOConv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    DOConv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    DOConv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
````
to apply DO-Conv without any other changes. 
````
python sample_pt.py
````
3 . We provide the performance improvement in this demo example as follows. (averaged accuracy (%) of five runs)

|          | run1  | run2  | run3  | run4  | run5  | avg    | +     |
|----------|-------|-------|-------|-------|-------|--------|-------|
| Baseline | 94.63  | 95.31 | 95.23 | 95.24 | 95.37 | 95.156 | -     |
| DO-Conv  | 95.59 | 95.73 | 95.68 | 95.70 | 95.67 | 95.674 | 0.518 |

4 . Then you can use DO-Conv in your own network in this way.

## Example Usage: GluonCV (mxnet-cu100==1.5.1.post0, gluoncv==0.6.0)
We show how to use DO-Conv based on the examples provided in the <a href="https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/packages/gluon/image/mnist.html" target="_blank">Tutorial</a> of GluonCV with MNIST dataset.

1 . Run the demo example first to get the accuracy of the baseline.
````
python sample_gluoncv.py
````
If there is any wrong at this step, please check whether the mxnet and gluoncv versions meets the requirements.

2 . Replace these lines:
````
self.conv1 = Conv2D(20, kernel_size=(5,5))
self.conv2 = Conv2D(50, kernel_size=(5,5))
````
with
````
self.conv1 = DOConv2D(1, 20, kernel_size=(5, 5))
self.conv2 = DOConv2D(20, 50, kernel_size=(5, 5))
````
to apply DO-Conv, note that the 'in_channels' in DOConv2D of GluonCV should be set explicitly. 
````
python sample_gluoncv.py
````
3 . We provide the performance improvement in this demo example as follows. (averaged accuracy (%) of five runs)

|          | run1  | run2  | run3  | run4  | run5  | avg    | +     |
|----------|-------|-------|-------|-------|-------|--------|-------|
| Baseline | 98.10 | 98.10 | 98.10 | 98.10 | 98.10 | 98.10 | -     |
| DO-Conv  | 98.26 | 98.26 | 98.26 | 98.26 | 98.26 | 98.26 | 0.16 |

4 . Then you can use DO-Conv in your own network in this way.



