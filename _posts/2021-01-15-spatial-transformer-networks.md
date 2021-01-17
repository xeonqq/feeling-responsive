---
layout: page
title: "Hands-on: implement a spatial transformer network by yourself"
subheadline: "Machine Learning"
teaser: "Spatial transformer networks understand how to rotate, translate a image. How does it learn to do it?"
header: no
sidebar: right
image:
    title: stn.png
    thumb:  stn.png
    homepage: stn.png
    caption: Spatial Transformer Networks
comments: true
categories:
    - machine learning
---
<style>
* {
  box-sizing: border-box;
}

.img-container {
  float: left;
  width: 50%;
  padding: 5px;
}

.clearfix::after {
  content: "";
  clear: both;
  display: table;
}
</style>

### Intro
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) is a paper published by *Max Jaderberg, Karen Simonyan, Andrew Zisserman* and *Koray Kavukcuoglu* at 2015, at the moment of writing, it has been cited 3715 times. 
The network works like a plug and play enhancement NN module to your exisiting neural network. It can perform affine transformation to the input image in order to ensure the inputs to the classification network to be spatial invariant.

The figure below shows how a STN can transform an affine distorted MNIST dataset into a "corrected" one:
[![stn in action][6]][6]


##### Why spatial invariant is important and what does that mean?

Let's look at the following examples using this [online hand-written digit recognition tool][2] trained with MNIST dataset. 

First we draw a normal 2, it has been recognized correctly, second we draw a rotated 2, it's recognized as 8:

<div class="clearfix">
  <div class="img-container">
  <img src="{{ site.urlimg }}normal_2.gif" alt="normal" style="width:100%">
  </div>
  <div class="img-container">
  <img src="{{ site.urlimg }}rotated_2.gif" alt="rotated" style="width:100%">
  </div>
</div>


##### One might say, because the network is not trained with such rotated inputs, therefore the network fails to recognize in this case. 

To verify it, I build a [simple network][4] with two fully connected layer same as the [google tutorial][3]. Then I trained this network with both original MNIST dataset and affine distorted MNIST dataset mixed together. 
The accuracy when evaluating the original MNIST is still good, with accuracy of 0.9767. However, when evaluating with distorted MNIST, the accuracy has only **0.7569**. 
In this case, we can say this fully connected layer based model architecture is not spatial invariant, because it fails to classify the images when they are affine transformed (scale, rotate, translation, etc).

##### Well, how about CNN with max-pooling layers. Because I hear from [here][5] that max-pooling layers help to remove spatial variance.

Talking is cheap, let also experiment a CNN with distorted MNIST dataset. Below is the construction of the CNN, with 2 conv layers + 2 max pooling layers:
``` python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), input_shape=(H, W,1),padding='valid',activation="relu"),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(16, (3,3),padding='valid',activation="relu"),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
])
```
After training with the same dataset as before, we have an accuracy of **0.88** for the distorted MNIST dataset! That's already a lot of improvement. Therefore, it is right indeed that max-pooling layer helps to remove the spatial variance
of the input.

#### Can we do better by using STN
In this post, we will see how Spatial Transformer Networks can allievate spatial variance problem, and how to implement the STN concept using keras from tensorflow 2. We will also cover many important details during implementation. 


### Implementation
*The full code for this project is available on GitHub:*
[https://github.com/xeonqq/spatial_transformer_network][1]

[1]: https://github.com/xeonqq/spatial_transformer_network
[2]: https://www.cs.ryerson.ca/~aharley/vis/conv/
[3]: https://www.tensorflow.org/tutorials/quickstart/beginner
[4]: https://github.com/xeonqq/spatial_transformer_network/blob/master/mnist_baseline.py
[5]: https://datascience.stackexchange.com/questions/35836/does-pooling-remove-spatial-information-of-image-in-cnn
[6]: https://github.com/xeonqq/spatial_transformer_network/blob/master/pics/distorted_and_undistorted_mnist.png?raw=true
