# tiny-yolo-noBatch

 tiny yolo v2 trained on the cocodataset 
 
 converted for tfjs  /Inference only
 
 
 
 cfg  : https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg
 weights: https://pjreddie.com/media/files/yolov2-tiny.weights
 no batch : 
 
Batch normalization is important when training a deep network, but it turns out we can get rid of it at inference time.
Which is a good thing because not having to do the batch norm calculations will make the whole faster.
Batch normalization usually happens after the convolutional layer but before the activation function gets applied (a so-called “leaky” ReLU in the case of YOLO).
Since both convolution and batch norm perform a linear transformation of the data, we can combine the batch normalization layer’s parameters with the weights for the convolution.
This is called “folding” the batch norm layer into the convolution layer.
Long story short, with a bit of math we can get rid of the batch normalization layers but it does mean we have to change the weights of the preceding convolution layer.

A quick recap of what a convolution layer calculates: if x is the pixels in the input image and w is the weights for the layer,
then the convolution basically computes the following for each output pixel:
 ``` 
out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2] + ... + x[i+k]*w[k] + b
 ```
This is a dot product of the input pixels with the weights of the convolution kernel, plus a bias value b.

And here’s the calculation performed by the batch normalization to the output of that convolution:
 ```
        gamma * (out[j] - mean)
bn[j] = ---------------------- + beta
            sqrt(variance)
 ```
It subtracts the mean from the output pixel, divides by the variance, multiplies by a scaling factor gamma, and adds the offset beta.
These four parameters — mean, variance, gamma, and beta — are what the batch normalization layer learns as the network is trained.

To get rid of the batch normalization, we can shuffle these two equations around a bit to compute new weights and bias terms for the convolution layer:
 ```
           gamma * w
w_new = --------------
        sqrt(variance)

        gamma*(b - mean)
b_new = ---------------- + beta
         sqrt(variance)
 ```
Performing a convolution with these new weights and bias terms on input x will give the same result as the original convolution plus batch normalization.
Now we can remove this batch normalization layer and just use the convolutional layer, but with these adjusted weights and bias terms w_new and b_new.
We repeat this procedure for all the convolutional layers in the network.

Note: The convolution layers in YOLO don’t actually use bias, so b is zero in the above equation. 
But note that after folding the batch norm parameters, the convolution layers do get a bias term.

src : [Here](http://machinethink.net/blog/object-detection-with-yolo/)
