<h1>Flattening</h1>


ANN only works with 1D input vectors — simple lists of numbers.

But images are 2D or 3D arrays (width × height × color channels).

So, before feeding an image into a normal ANN, we must flatten it.

Example:
If an image is 28×28 pixels (like MNIST digits), it has
28 × 28 = 784 features.
We convert the image into a 784×1 vector using a Flatten layer.

 Flatten layer → turns multi-dimensional input (e.g. 28×28) into 1D vector (784) but a lot of spetial information is lost and the relative relation is lost . 

<h1>CNN</h1>

-> Downsample the original input but maintain the spetial information


Then CNN (Convolutional Neural Network) came (1stly used in 1980 but named in 1989 by Ly Chy)

CNNs were designed especially for images, to preserve spatial structure (relationships between neighboring pixels).

CNNs don’t flatten the image at the beginning.

Instead, they use:

Convolution layers → detect features (edges, textures)

Pooling layers → reduce size but keep spatial info

Then finally, before the fully connected (Dense) layer at the end, they Flatten the feature maps.


<h3>1. Grayscale Images</h3>

These have only one color channel (intensity of light).

Each pixel stores a single value (brightness) between 0 and 255.

Example:

A 28×28 grayscale image → shape = (28, 28, 1)

(the last 1 is the number of channels)

 Example pixel:

Pixel value = 0 → black  
Pixel value = 255 → white  


<h3> 2. Colored Images </h3>

These use 3 channels: Red, Green, Blue (RGB).

Each pixel is made of 3 numbers — one for each color component.

Example:

A 64×64 color image → shape = (64, 64, 3)

(the last 3 shows RGB channels)

Example pixel:

[255, 0, 0] → pure red  
[0, 255, 0] → pure green  
[0, 0, 255] → pure blue  

