# DO-Conv: Depthwise Over-parameterized Convolutional Layer

Created by Jinming Cao, <a href="http://yangyan.li" target="_blank">Yangyan Li</a>, Mingchao Sun, <a href="https://scholar.google.com/citations?user=NpTmcKEAAAAJ&hl=en" target="_blank">Ying Chen</a>, <a href="https://www.cs.huji.ac.il/~danix/" target="_blank">Dani Lischinski</a>, <a href="https://danielcohenor.com/" target="_blank">Daniel Cohen-Or</a>, <a href="https://cfcs.pku.edu.cn/baoquan/" target="_blank">Baoquan Chen</a>, and <a href="http://irc.cs.sdu.edu.cn/~chtu/index.html" target="_blank">Changhe Tu</a>.

## Introduction

DO-Conv is a depthwise over-parameterized convolutional layer, which can be used as a replacement of conventional convolutional layer in CNNs in the training phase to achieve higher accuracies. In the inference phase, DO-Conv can be fused into a conventional convolutional layer, resulting in the computation amount that is exactly the same as that of a conventional convolutional layer.

Please see our <a href="https://arxiv.org/abs/2006.12030" target="_blank">preprint on arXiv</a> for more details, where we demonstrated the advantages of DO-Conv on various benchmark datasets/tasks.

## ImageNet Classification Performance

We take the model zoo of <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html" target="_blank">GluonCV</a> as baselines. The settings in the baselines have been tuned to favor baselines, and they are not touched during the switch to DO-Conv. In other words, DO-Conv is the one and only change over baselines, and no hyper-parameter tuning is conducted to favor DO-Conv. We consider GluonCV highly reproducible, but still, to exclude clutter factors as much as possible, we train the baselines ourselves, and compare DO-Conv versions with them, while reporting the performance provided by GluonCV as reference. The results are summarized in this table where the “DO-Conv” column shows the performance gain over the baselines.
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
    <td>ResNeXt</td>
    <td>50_32x4d</td>
    <td>79.32</td>
    <td>79.21</td>
    <td>+0.40</td>
  </tr>
</tbody>
</table>

## DO-Conv Usage

In thie repo, we provide reference implementation of DO-Conv in Tensorflow, PyTorch and GluonCV, as replacement to tf.keras.layers.Conv2D, torch.nn.Conv2d and mxnet.gluon.nn.Conv2D, respectively. Please see the code for more details.
