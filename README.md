# Computer Vision Lab: Week Experiments

Setup: `pip install scikit-image opencv-python scikit-learn scipy matplotlib`

## 1. LBP texture and cascade face detection

Computes a Local Binary Pattern map and detects the astronaut's face with a skimage cascade.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import data, color
from skimage.feature import Cascade, local_binary_pattern

img = data.astronaut()
gray = color.rgb2gray(img)
lbp = local_binary_pattern(gray, P=16, R=2)

detector = Cascade(data.lbp_frontal_face_cascade_filename())
lbp_cascade = detector.detect_multi_scale(img,
                                         scale_factor=1.2,
                                         step_ratio=1,
                                         min_size=(60, 60),
                                         max_size=img.shape[:2])

plt.figure()
plt.imshow(img)
ax = plt.gca()
for face in lbp_cascade:
    rect = Rectangle((face['c'], face['r']),
                     face['width'],
                     face['height'],
                     edgecolor='r',
                     facecolor='none',
                     linewidth=2)
    ax.add_patch(rect)
plt.show()

# Second image: LBP
plt.figure()
plt.imshow(lbp, cmap='gray')
plt.axis('off')
plt.show()
```

## 2. Bag of visual words (SIFT + K-Means)

Clusters SIFT descriptors from both images into 32 visual words, then plots each image as a word histogram.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data
import cv2

images = {
  "astronaut": data.astronaut(),
  "cat": data.chelsea()
}

sift = cv2.SIFT_create()

all_desc = []
image_desc = {}

for name, img in images.items():
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  kp, desc = sift.detectAndCompute(gray, None)
  all_desc.append(desc)
  image_desc[name] = desc

all_desc = np.vstack(all_desc)

kmeans = KMeans(n_clusters=32)
kmeans.fit(all_desc)

histograms = {}

for name, desc in image_desc.items():
  labels = kmeans.predict(desc)
  hist = np.bincount(labels, minlength=32)
  histograms[name] = hist / np.sum(hist)

for i, (name, hist) in enumerate(histograms.items()):
  plt.bar(range(32), hist)
  plt.title(name)
  plt.xlabel("Cluster")
  plt.ylabel("Frequency")
  plt.show()
```

Note: the original cell was missing `import numpy as np`, which is added here.

## 3. Image stitching

Stitches the left and right stereo motorcycle images into one panorama.

```python
import cv2
import matplotlib.pyplot as plt
from skimage import data

left, right, _ = data.stereo_motorcycle()
stitcher = cv2.Stitcher_create()
status, panorama = stitcher.stitch([left, right])

if status == cv2.Stitcher_OK:
  plt.imshow(panorama)
  plt.show()
```

## 4. SIFT keypoints

Draws the detected SIFT keypoints on both sample images.

```python
import matplotlib.pyplot as plt
from skimage import data
import cv2

images = {
  "astronaut": data.astronaut(),
  "cat": data.chelsea()
}

sift = cv2.SIFT_create()

for name, img in images.items():
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  kp, desc = sift.detectAndCompute(gray, None)

  # Draw detected keypoints on the image
  img_with_keypoints = cv2.drawKeypoints(img, kp, None)

  plt.figure(figsize=(8, 6))
  plt.imshow(img_with_keypoints)
  plt.title(f"SIFT Keypoints for {name.strip()}")
  plt.axis('off')
  plt.show()
```

## 5. HOG features

Extracts Histogram of Oriented Gradients features and visualizes them.

```python
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import hog

# Load a sample image containing a person
img = data.astronaut()

# Convert RGB image to grayscale
gray = color.rgb2gray(img)

# Extract HOG features
features, hog_img = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

print("HOG Feature Length:", len(features))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_img, cmap='gray')
plt.title("HOG Features")
plt.axis("off")

plt.tight_layout()
plt.show()
```

Output:

```text
HOG Feature Length: 142884
```

## 6. Canny edge detection

Blurs the image to reduce noise, then finds edges with thresholds 100 and 200.

```python
import cv2
import matplotlib.pyplot as plt
from skimage import data

img = data.camera()  # already a grayscale image

# Reduce noise before finding edges
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Detect edges
edges = cv2.Canny(blur, 100, 200)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blur, cmap="gray")
plt.title("Blurred")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.tight_layout()
plt.show()
```

## 7. Morphological operations

Applies erosion, dilation, opening and closing to a thresholded binary image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = data.chelsea()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Convert to a binary image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Create a 5 x 5 kernel
kernel = np.ones((5, 5), np.uint8)

# Morphological operations
erosion = cv2.erode(binary, kernel)
dilation = cv2.dilate(binary, kernel)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

images = [binary, erosion, dilation, opening, closing]
titles = ["Binary", "Erosion", "Dilation", "Opening", "Closing"]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## 8. JPEG compression

Saves the image at quality 10, 50 and 90 and compares file sizes.

```python
import cv2
import os
import matplotlib.pyplot as plt
from skimage import data

img = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)

# Save the original image
cv2.imwrite("original.jpg", img)
original_size = os.path.getsize("original.jpg")
print("Original Size:", original_size)

qualities = [10, 50, 90]

for i, q in enumerate(qualities):
    filename = f"compressed_{q}.jpg"
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])

    size = os.path.getsize(filename)
    compressed = cv2.imread(filename)

    print("Quality:", q, "Size:", size)

    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    plt.title("Quality " + str(q))
    plt.axis("off")

plt.tight_layout()
plt.show()
```

Output (bytes):

```text
Original Size: 50163
Quality: 10 Size: 5291
Quality: 50 Size: 13773
Quality: 90 Size: 35042
```

Lower quality gives a much smaller file at the cost of visible artifacts.

## 9. Thresholding and K-Means segmentation

Compares a global threshold with K-Means colour clustering (3 clusters).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from sklearn.cluster import KMeans

# Load image
img = data.chelsea()

# Convert to grayscale and apply thresholding
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Apply K-Means clustering
pixels = img.reshape(-1, 3)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(pixels)

segmented = kmeans.cluster_centers_[labels]
segmented = segmented.reshape(img.shape).astype(np.uint8)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented)
plt.title("K-Means")
plt.axis("off")

plt.show()
```

## 10. Smoothing filters

Compares box, Gaussian, median and bilateral filters.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)

# Linear filter: box/averaging filter
kernel = np.ones((7, 7), np.float32) / 49
box = cv2.filter2D(img, -1, kernel)

# Non-linear / smoothing filters
gaussian = cv2.GaussianBlur(img, (7, 7), 0)
median = cv2.medianBlur(img, 7)
bilateral = cv2.bilateralFilter(img, 11, 60, 60)

images = [img, box, gaussian, median, bilateral]
titles = ["Original", "Box Filter", "Gaussian", "Median", "Bilateral"]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    result = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## 11. Histogram equalization and mean/Gaussian blur

Improves contrast on a grayscale image and compares two smoothing filters.

```python
import cv2
import matplotlib.pyplot as plt
from skimage import data

img = data.chelsea()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Improve contrast
hist = cv2.equalizeHist(gray)

# Smooth the image
mean = cv2.blur(gray, (5, 5))
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

images = [gray, hist, mean, gaussian]
titles = ["Grayscale", "Histogram Equalization", "Mean Filter", "Gaussian Filter"]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## 12. Geometric transformations

Translation, rotation, scaling and shearing with affine matrices.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = data.chelsea()
h, w = img.shape[:2]

# 1. Translation: move right by 100 and down by 50 pixels
M = np.float32([[1, 0, 100], [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (w, h))

# 2. Rotation: rotate by 45 degrees
M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1)
rotated = cv2.warpAffine(img, M, (w, h))

# 3. Scaling: increase width and height by 1.5 times
scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

# 4. Shearing: slant the image
M = np.float32([[1, 0.2, 0], [0, 1, 0]])
sheared = cv2.warpAffine(img, M, (w, h))

images = [img, shifted, rotated, scaled, sheared]
titles = ["Original", "Shifted", "Rotated", "Scaled", "Sheared"]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
```

## 13. Image properties and negative

Prints shape, pixel range and channel means, then builds a negative image.

```python
import cv2
from skimage import data
import matplotlib.pyplot as plt

img = data.chelsea()  # sample cat image in RGB format

h, w, c = img.shape
print("Height:", h)
print("Width:", w)
print("Channels:", c)
print("Aspect Ratio:", w / h)
print("Pixel Range:", img.min(), "to", img.max())

r, g, b = cv2.split(img)
print("Red Mean:", r.mean())
print("Green Mean:", g.mean())
print("Blue Mean:", b.mean())

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
negative = 255 - gray

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(negative, cmap="gray")
plt.title("Negative Image")
plt.axis("off")

plt.show()
```

Output:

```text
Height: 300
Width: 451
Channels: 3
Aspect Ratio: 1.5033333333333334
Pixel Range: 0 to 231
Red Mean: 147.67308943089432
Green Mean: 111.44447893569844
Blue Mean: 86.79785661492978
```

## 14. Matrix operations with SciPy

Basic matrix arithmetic, determinant, LU, SVD and eigen decomposition. The matrix is singular, so the determinant is about 0.

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

print("Matrix A:\n", A)
print("Matrix B:\n", B)

print("Addition:\n", A + B)
print("Subtraction:\n", A - B)
print("Element-wise multiplication:\n", A * B)
print("Matrix multiplication:\n", A @ B)
print("Transpose:\n", A.T)

print("Determinant:", linalg.det(A))

lu, piv = linalg.lu_factor(A)
print("LU factor matrix:\n", lu)
print("Pivot indices:", piv)

U, S, Vt = linalg.svd(A)
print("Singular values:", S)

values, vectors = linalg.eig(A)
print("Eigenvalues:\n", values)
print("Eigenvectors:\n", vectors)
```