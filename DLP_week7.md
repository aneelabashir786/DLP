# Flattening

ANN only works with **1D input vectors** — simple lists of numbers.
But images are **2D or 3D arrays** (width × height × color channels).

Before feeding an image into a normal ANN, we must **flatten** it.

**Example:**
If an image is 28×28 pixels (like MNIST digits), it has
28 × 28 = **784 features**.
We convert the image into a **784×1 vector** using a Flatten layer.

The Flatten layer turns multi-dimensional input (e.g., 28×28) into a 1D vector (784).
However, a lot of **spatial information is lost**, and the **relative relationship between pixels** disappears.

---

# CNN (Convolutional Neural Networks)

CNNs were designed **especially for images** to preserve **spatial structure** — relationships between neighboring pixels.
They were first used in 1980 and officially named in **1989 by Yann LeCun**.

### Key Idea

CNNs downsample the original input but maintain spatial information.

CNNs don’t flatten the image at the beginning.
Instead, they use:

* **Convolution layers** → detect features (edges, textures)
* **Pooling layers** → reduce size but keep spatial info
* **Flattening** → done after convolution and pooling, before fully connected (Dense) layers

---

## 1. Grayscale Images

* Have **one color channel** (intensity of light).
* Each pixel stores a single value (brightness) between 0 and 255.
* Example shape: **(28, 28, 1)**

| Pixel Value | Meaning |
| ----------- | ------- |
| 0           | Black   |
| 255         | White   |

---

## 2. Colored Images (RGB)

* Use **3 channels**: Red, Green, Blue (RGB).
* Each pixel = 3 numbers (one per channel).
* Example shape: **(64, 64, 3)**

| RGB Value   | Color |
| ----------- | ----- |
| [255, 0, 0] | Red   |
| [0, 255, 0] | Green |
| [0, 0, 255] | Blue  |

---

# Convolution Layer

## 1. What Is Feature Extraction?

Feature extraction means identifying important information from an image — such as edges, corners, shapes, and textures — that helps a model understand what the image contains.

### Traditional Feature Extraction Methods (Before CNNs)

Humans used handcrafted feature extraction methods such as:

* **HOG (Histogram of Oriented Gradients)**
* **LBP (Local Binary Patterns)**
* **Gabor Filters, SIFT, SURF**

These were manually designed to capture specific image characteristics.

---

### HOG (Histogram of Oriented Gradients)

* Detects edges and shapes in an image.
* Divides the image into small cells and computes gradient directions (how brightness changes).
* Builds a histogram of edge orientations.

**Output:** A feature vector describing edge directions, often used in human or object detection.

---

### LBP (Local Binary Pattern)

* Captures texture patterns.
* Compares each pixel with its neighbors:

  * If neighbor ≥ center → 1
  * Else → 0
* Generates binary patterns (e.g., 11010010), converts to decimals, and creates a histogram.

**Output:** A representation of texture, useful for face recognition or surface analysis.

---

### Gabor Filters and Similar Methods

* Capture frequency and orientation information.
* Detect textures and edges at various angles.

---

## 2. CNN Feature Extraction

CNNs automatically learn what kind of features to extract — no need for manual HOG, LBP, or Gabor filters.
Each convolutional layer learns its own filters to detect edges, corners, and complex shapes.

---

## 3. What Is a Convolution Layer?

A convolution layer is the main building block of a CNN.
It uses small filters (e.g., 3×3 or 5×5) that slide over the image to create feature maps.

Each filter automatically learns to detect:

* Edges
* Corners
* Textures
* Object parts
* Complex shapes (in deeper layers)

---

### Mathematical Formula (Stride = 1)

[
\text{Output Size} = N - K + 1
]

Where:

* ( N ) = Input size
* ( K ) = Filter (kernel) size

**Example (non-square image):**
Input = 32×30, Filter = 5×5
[
(32 - 5 + 1) × (30 - 5 + 1) = 28 × 26 × 1
]

---

# Padding

When a filter slides over an image, boundary pixels are not fully covered — so the output shrinks.

### Example

Input = 5×5
Filter = 3×3
Stride = 1

[
\text{Output Size} = \text{Input Size} - \text{Filter Size} + 1 = 5 - 3 + 1 = 3
]

So output = 3×3.

The larger the filter, the smaller the output feature map.

To prevent loss of image size, CNNs use **padding**.
Padding means adding extra rows/columns (usually zeros) around the image before applying the filter.

**Example:**
If you add 1-pixel zero-padding around a 5×5 image, it becomes 7×7.
Now applying a 3×3 filter gives:

[
7 - 3 + 1 = 5
]

Output size = same as input (5×5)

**General Formula:**

[
\text{Output Size} = N - K + 2P + 1
]

---

## 1. Effect on Time Complexity

Because the input image size becomes larger, the convolution operation does more multiplications and additions.
This increases computation time per layer.
Since CNNs have many layers, total training time increases.

[
\text{Time Complexity} \propto \text{Input Size} \times \text{Number of Filters}
]

If the input stays large due to padding, every filter performs more work.

---

## 2. Effect on Space (Memory) Complexity

Padding increases:

* Memory required for feature maps
* Intermediate storage
* GPU/CPU memory usage during forward and backward propagation

Thus, while padding keeps the output size constant, it increases memory and computational load.

---

# Stride

## 1. What Is Stride?

Stride (S) means how many pixels the filter moves at each step when sliding across the image.

* Stride = 1 → filter moves 1 pixel at a time
* Stride = 2 → filter jumps 2 pixels at a time

Stride controls how much overlap occurs between filter positions.

---

## 2. Formula (with Stride)

[
O = \frac{(W - F + 2P)}{S} + 1
]

From this formula:

If **S increases**, the denominator increases, causing output size to decrease.
Larger stride → smaller feature map.

---

# Learnable Parameters in a Convolution Layer

Each convolutional layer has filters (kernels) that learn weights.
The total number of learnable parameters includes both weights and bias terms.

[
\text{Parameters} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
]

Where:

* ( K_h ) = filter height
* ( K_w ) = filter width
* ( C_{in} ) = number of input channels
* ( C_{out} ) = number of output channels (number of filters)
* ( +1 ) = bias for each filter

---

# Multiply-Add (MAC) Operations

When a convolution filter slides over the image, at each position it:

* Multiplies every filter weight with the corresponding input pixel
* Adds all results to produce one output value

This is called a **Multiply–Accumulate (MAC)** operation.

---

## Formula to Calculate MAC Operations

For a single convolutional layer:

[
\text{Total MACs} = K_h \times K_w \times C_{in} \times H_{out} \times W_{out} \times C_{out}
]

Where:

* ( K_h, K_w ) = filter height and width
* ( C_{in} ) = number of input channels
* ( C_{out} ) = number of filters (output channels)
* ( H_{out}, W_{out} ) = output feature map height and width

---

**Notes:**

* Each MAC = 1 multiplication + 1 addition
* Some references express complexity in FLOPs (Floating Point Operations), approximately **2 × MACs**


# Non-linearity : 

ReLU + Tanh (in hidden layers ) 
sigmoid + softmax (on output layer )
