# Loopy Belief Propagation
Denoise a given binary image using Loopy Belief Propagation

Theory:
Loopy Belief Propagation is an algorithm for approximate inference in graphical models

The algorithm takes as input an n x n binary image (X) encoded by a 0-1 matrix. It returns another n x n matrix (Y) which is a denoised version of X

The assumptions is that a particular pixel value in the denoised image depends on:
    1) The value of the corresponding pixel in the input (noisy image) &
    2) The value of the neighboring pixels in the denoised image

Considering the above assumptions, a Probabilistic Graphical Model is formulated with pixels of input image and desired output image as nodes. Edges are established between all pairs of neighboring nodes of the input as well as the output image. Also, corresponding pixels of the input and output image are connected with an edge. Parameters $\theta$ and $\gamma$ determine how much the algorithm relies on each of the two assumptions mentioned above. 

Results:
![Results_Cameraman](https://github.com/sanjeevg15/loopy-bp-denoise/blob/master/images/cameraman.jpg)
![Results_Circle](https://github.com/sanjeevg15/loopy-bp-denoise/blob/master/images/circle.jpg)
