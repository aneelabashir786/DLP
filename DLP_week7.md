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


<h1>Convolution Layer</h1>

1. What is Feature Extraction?

Feature extraction means identifying important information from an image — such as edges, corners, shapes, and textures — that helps a model understand what the image contains.

Before CNNs were introduced, humans used handcrafted feature extraction methods, such as:

HOG (Histogram of Oriented Gradients)

LBP (Local Binary Patterns)

Gabor Filters, SIFT, or SURF

These were manually designed to capture specific image characteristics.

2. Traditional Feature Extraction Methods

    HOG (Histogram of Oriented Gradients)

Used to detect edges and shapes in an image.

Works by dividing the image into small regions (“cells”) and computing the gradient directions (how brightness changes).

Creates a histogram of edge orientations.

   Output: A feature vector that describes edge directions — often used in tasks like human detection or object recognition.

   LBP (Local Binary Pattern)

Used to capture texture patterns.

Compares each pixel with its neighboring pixels:

If a neighbor’s value ≥ center pixel → write “1”

Otherwise → write “0”

The result is a binary pattern (like 11010010) converted into a decimal value.

A histogram of these values gives a texture description.

 Output: A representation of texture, useful for face recognition or surface analysis.

   Gabor Filters and Similar Methods

Capture both frequency and orientation information.

Can detect textures and edges at various angles.

🤖 3. CNNs (Convolutional Neural Networks)

When CNNs were developed, they replaced these manual methods.
Now, CNNs automatically learn what kind of features to extract — no need for HOG, LBP, or Gabor filters manually.

⚙️ 4. What is a Convolution Layer?

A convolution layer is the main building block of a CNN.

It uses small filters (kernels), such as 3×3 or 5×5 matrices.

These filters slide over the image, performing a convolution operation.

The result is a feature map, which highlights specific patterns.

Each filter automatically learns to detect:

Edges

Corners

Textures

Object parts

Complex shapes (in deeper layers)

🔹 How It Works Mathematically

( with 1 stride )
           
           output size = N - k + 1 

N = size of input image 
K = size of filter 

for Non squared image : 

32 x 30 

(32 - 5 + 1 ) x (30 - 5 + 1) = 28 x 26 x 1(no of filters ) 


<h1>Padding</h1>

When a filter (kernel) slides over an image during convolution, it cannot cover the boundary pixels completely — so the output size shrinks.

Let’s take an example:

Suppose input image size = 5×5

Filter (kernel) size = 3×3

Stride = 1

Then the output feature map size is calculated by the formula:

Output size  = Input size − Filter size + 1

   =  5    −   3  +   1  =  3
   
 So output = 3×3

**That means — the larger the filter, the smaller the output feature map.**


To prevent loss of image size, CNNs use padding.

Padding means adding extra rows/columns (usually zeros) around the border of the input image before applying the filter.

Example:

If you add 1 pixel of zero-padding around a 5×5 image → it becomes 7×7.
Now applying a 3×3 filter gives:

7  −  3 +  1  =  5

Output size = same as input (5×5)

Mathematically : 

         output size = N−K+2P​+1


<h3>1. Effect on Time Complexity</h3>

Because the input image size is now larger:

The convolution operation has to do more multiplications and additions.

That increases the computation time per layer.

And since CNNs have many layers, total training time increases.

In short:

Time Complexity
∝
Input Size
×
Number of Filters
Time Complexity∝Input Size×Number of Filters

So if the input stays large due to padding, every filter does more work.

 <h3>2. Effect on Space (Memory) Complexity</h3>

Padding also increases:

The memory needed to store intermediate feature maps.

The number of parameters (indirectly, if fully connected layers follow large outputs).

The GPU/CPU memory usage during forward and backward propagation.

So yes — while padding keeps the feature map size constant, it also increases the total memory and computation load.


<h1>Stride</h1>

1. What Is Stride?

Stride (S) means how many pixels the filter (kernel) moves at each step when sliding across the image.

Stride = 1 → the filter moves 1 pixel at a time (dense scanning)

Stride = 2 → the filter jumps 2 pixels at a time (skips some pixels)

So stride controls how much overlap happens between filter positions.

2. Formula (with Stride)

You already know the formula:

      𝑂  =  (𝑊 − 𝐹 + 2𝑃)/𝑆 + 1
​

From this, you can see:

If S increases → denominator increases → Output size decreases . That means larger stride → smaller feature map


1. Formula for Learnable Parameters in a Convolution Layer

Each convolutional layer has filters (kernels) that learn weights.
The total number of learnable parameters includes:

The weights inside each filter

The bias term (optional, but usually included)

So, the formula is:


    Parameters =  ( 𝐾ℎ  ×  𝐾𝑤  ×  𝐶𝑖𝑛  +  1) ×  𝐶𝑜𝑢𝑡

	​Where:

        𝐾ℎ = filter (kernel) height

        𝐾𝑤 = filter width

        𝐶i𝑛 = number of input channels

       𝐶𝑜𝑢𝑡 = number of output channels (number of filters)

	   +1 = bias term for each filter
