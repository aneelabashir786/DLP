<h1>Flatten</h1>


ANN only works with 1D input vectors — simple lists of numbers.

But images are 2D or 3D arrays (width × height × color channels).

So, before feeding an image into a normal ANN, we must flatten it.

Example:
If an image is 28×28 pixels (like MNIST digits), it has
28 × 28 = 784 features.
We convert the image into a 784×1 vector using a Flatten layer.

 Flatten layer → turns multi-dimensional input (e.g. 28×28) into 1D vector (784) but a lot of special information is lost and the relative relation is lost . 
