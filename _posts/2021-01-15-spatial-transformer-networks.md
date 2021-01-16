---
layout: page
title: "Hands-on: implement a spatial transformer network by yourself"
subheadline: "Machine Learning"
teaser: "Spatial transformer networks understand how to rotate, translate a image. How does it learn to do it?"
header: no
image:
    title: stn.png
    thumb:  stn.png
    homepage: stn.png
    caption: Spatial Transformer Networks
comments: true
categories:
    - machine learning
---

### Intro
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) is a paper published by *Max Jaderberg, Karen Simonyan, Andrew Zisserman* and *Koray Kavukcuoglu* at 2015, at the moment of writing, it has been cited 3715 times. 
The network works like a plug and play enhancement NN module to your exisiting neural network. It can perform affine transformation to the input image in order to ensure the inputs to the classification network to be spatial invariant.

Why spatial invariant is important and what does that mean?
Let's look at the following examples using this [online hand-written digit recognition tool][2] trained with MNIST dataset. 

First we draw a normal 2, it has been recognized correctly:

<img src="{{ site.urlimg }}normal_2.gif" alt="">

Second we draw a rotated 2, it's recognized as 8:

<img src="{{ site.urlimg }}rotated_2.gif" alt="">

Because the network is not trained with such rotated inputs, therefore the network fails to recognize in this case. We call this is not spatial invariant. 

In this post, we will implement the STN concept using keras from tensorflow 2. We will cover many important details during implementation. 

*The full code for this project is available on GitHub:*
[https://github.com/xeonqq/spatial_transformer_network][1]

### Implementation

[1]: https://github.com/xeonqq/spatial_transformer_network
[2]: https://www.cs.ryerson.ca/~aharley/vis/conv/

