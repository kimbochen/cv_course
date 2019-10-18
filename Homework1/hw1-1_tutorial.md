# CV HW 1 Tutorial: Harris Corner

## Requirements

```
python 3.6+
numpy
scipy
scikit-image
matplotlib
```

`numpy` and `scipy` are used for basic operations. `scikit-image` (`skimage`) is used for image stuff. `matplotlib` is the de-facto visualiation tool in `python`.

You should use **virtual environment** or Colab to isolate this project from other python project or  system's python to prevent pollution. Code in this project utilizes the mechanism of `ndarray`, so you should have read [Numpy Quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html) before, especially **indexing**, **masking**, **broadcasting**. If you got time, [Numpy Indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html) is also a good material.

The `import` in this tutorial is:

```python=
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
```

## Gaussian Kernel

![](https://i.imgur.com/t5Vj3JE.png)

Take 5x5 gaussian kernel for example, we want to find the gaussian value of *black dots* and organize them in a 5x5 `ndarray`.

```python=
s = 5 # sigma

# Create the x, y coordinates of each dot
cx, cy = (0 + 5 - 1) / 2, (0 + 5 - 1) / 2 # shifting value
xx, yy = np.meshgrid(np.arange(5) - cx, np.arange(5) - cy)
# >>> print(xx)
# [[-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]]
# >>> print(yy)
# [[-2. -2. -2. -2. -2.]
#  [-1. -1. -1. -1. -1.]
#  [ 0.  0.  0.  0.  0.]
#  [ 1.  1.  1.  1.  1.]
#  [ 2.  2.  2.  2.  2.]]

# Calcuate the gaussian value of each dot
g = np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 2)
# Normalize the kernel to have sum = 1.0
g = g / g.sum()
# >>> print(g)
# [[0.03688345 0.03916419 0.03995536 0.03916419 0.03688345]
#  [0.03916419 0.04158597 0.04242606 0.04158597 0.03916419]
#  [0.03995536 0.04242606 0.04328312 0.04242606 0.03995536]
#  [0.03916419 0.04158597 0.04242606 0.04158597 0.03916419]
#  [0.03688345 0.03916419 0.03995536 0.03916419 0.03688345]]
```

Notice that `xx`, `yy`, `g` are `ndarray` with shape `[5, 5]`. The `np.exp(), **, *, /` operations are all element-wise. To do a gaussian blur, you just need to call `output = ndi.convole(img, g)`. 

Other way to obtain a 2D gaussian kernel is computing the outer product of two 1D Gaussian kernel using formula:

$$
G(x, y) = 
(\frac{1}{\sqrt{2\pi} \sigma_x}
exp(-\frac{x^2}{2 \sigma_x^2}))
((\frac{1}{\sqrt{2\pi} \sigma_y})
exp(-\frac{y^2}{2 \sigma_y^2}))
$$

```python=
sx, sy = 5, 5 # sigma
xx = np.arange(k) - (0 + k - 1) / 2 # [k,]
yy = np.arange(k) - (0 + k - 1) / 2 # [k,]
gx = np.exp(-0.5 * (xx / sx)**2) / (np.sqrt(2 * np.pi) * sx) # [k,]
gy = np.exp(-0.5 * (yy / sy)**2) / (np.sqrt(2 * np.pi) * sy) # [k,]
g = np.outer(gy, gx) # [k, k]
g = g / g.sum()
```

## Sobel

Assume you get the x-gradient `dx` and y-graident `dy`, the magnitue is simply:

```python=
mag = np.sqrt(dx ** 2 + dy ** 2)
```
What else can you expect? ``¯\_(ツ)_/¯``


To visualize gradient direction, [HSV colorspace](https://zh.wikipedia.org/zh-tw/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4) is a good way. Let *hue* represent the angle of the vector, so different angle has different color. Let *value* represent the magnitude of the vector, so the smaller the magnitude, the darker the color. *Saturation* can be a constant value like `1.0`. Since most visualization library do not support viewing hsv image, you'll need to convert hsv image back to rgb image.

![](https://i.imgur.com/6L8YqWX.png =300x)

For implementation, angle can be calculated using `np.arctan2`. Be aware that we are not using `np.arctan` since [atan2](https://en.wikipedia.org/wiki/Atan2) has some advantages (say, no need to deal with division by zero). Notice that  `skimage.color.hsv2rgb` expects hsv being represnted as `ndarray` with range `[0, 1]` and shape `[h, w, 3]`. You need some normalization to deal with the data range.

```python=
h, w = gray.shape
hsv = np.zeros((h, w, 3))
hsv[..., 0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
hsv[..., 1] = np.ones((h, w)) # or just write = 1.0
hsv[..., 2] = (mag - mag.min()) / (mag.max() - mag.min())
rgb = color.hsv2rgb(hsv)
```

Expected result (10x10 Gaussian):

![](https://i.imgur.com/zrnSySs.png)

## Structure Tensor

Structure tensor is calculated on each pixel of image from its local window. Each pixel has a `2x2` matrix and we'll find the determinant and trace of such matrix to derive harris corner response. Formally,

The structure tensor $A$ of position $(r, c)$ consists of weighted sums of the local patch of $I_{xx}(r, c), I_{xy}(r, c), I_{yy}(r, c)$: 

$$
A(r, c) = 
\begin{bmatrix}
\sum_{u, v} w(u,v) I_{xx}(r + u, c + v) &
\sum_{u, v} w(u,v) I_{xy}(r + u, c + v)
\\
\sum_{u, v} w(u,v) I_{yx}(r + u, c + v) & 
\sum_{u, v} w(u,v) I_{yy}(r + u, c + v)
\end{bmatrix}
$$

which can be rewritten using convolution:

$$
\begin{align}
A(r, c) 
=&
\begin{bmatrix}
A_{xx}(r, c) & A_{xy}(r, c) \\
A_{xy}(r, c) & A_{yy}(r, c)
\end{bmatrix} , where \\
& A_{xx} = I_{xx} * w \\
& A_{xy} = I_{xy} * w \\
& A_{yy} = I_{yy} * w
\end{align}
$$

Furthermore, by assuming $w$ is a Gaussian Kernel, we get:

$$
A_{xx} = GaussianBlur(I_{xx}) \\
A_{xy} = GaussianBlur(I_{xy}) \\
A_{yy} = GaussianBlur(I_{yy})
$$

My implmentation:

```python=
def structure_matrix(dx, dy):
    '''
    Args:
        dx: (ndarray) sized [H, W]
        dy: (ndarray) sized [H, W]
    Return:
        Axx: (ndarray) sized [H, W]
        Axy: (ndarray) sized [H, W]
        Ayy: (ndarray) sized [H, W]
        where structure tensor A of (r, c) is
        [
            [ Axx[r, c], Axy[r, c] ],
            [ Axy[r, c], Ayy[r, c] ]
        ]
    '''
    Axx = gaussian_blur(dx * dx, 15)
    Axy = gaussian_blur(dx * dy, 15)
    Ayy = gaussian_blur(dy * dy, 15)
    return Axx, Axy, Ayy
```

## Harris Corner Response

The response of pixel $(r, c)$ is $det(A(r, c)) - k \cdot tr(A(r, c))^2$ where 

1. $det(A(r, c))$ is `Axx[r, c] * Ayy[r, c] - Axy[r, c] * Axy[r, c]`
2. $tr(A(r, c))$ is `Axx[r, c] + Ayy[r, c]`

Like before, we utilize `ndarray` element-wise operations to obtain response. All pixels' response can be calculated together:

```python=
det = Axx * Ayy - Axy * Axy
tr = Axx + Ayy
R = det - k * tr * tr
```

As you expect, `det`, `tr`, `R` are `ndarray` with shape same as (gray) image.


## NMS

There's a lot of ways to do nms, the way I gonna use here is a simplified version of `skimage.feature.corner_peaks` which utilizes [maximum filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html). 

![](https://i.imgur.com/HI40o0d.png)


Pixel `(r, c)` is a harris corner if and only if:

1. `R[r, c] > threshold`. Threshold is tunned manually, something like `3e-7`.
2. `(r, c)` is a local maximum in `R`, that is, larger than the surrounding pixels' response.

(1.) is easy. `R > 3e-6` do the job. You'll get a `ndarray` with `dtype` `bool` representing the mask, i.e., whether each pixel satisfy the condition.
(2.) can be done using the fact that a local maximum remains same value after `R` being maximum filtered. Comparing the original response with maximum filtered one, the pixels which have same value in both should be a local maximum. `Scipy` has maximum-filter built-in, so `np.abs(ndi.maximum_filer(R) - R) < 1e-15` gets the mask of condition (2.). 

Finally, the mask that satisfies both condiations is combined from 2 masks:

```python=
mask1 = (R > 3e-6)
mask2 = (np.abs(ndi.maximum_filter(R, size=30) - R) < 1e-15)
mask = (mask1 & mask2) # or np.logical_and(...)
```

Notice that I don't compare `maxR` and `R` by `maxR == R` because they are both `float ndarray`. In general, doing comparision in floating points should not use `==` due to floating point error.

If you want to get the positions of the corners instead of the mask, `np.nonzero` extracts the positions of the positive elements of a mask.

![](https://i.imgur.com/pmi51U1.png)


## Misc.

### Reading Image

Images should be processed as `np.float32 ndarray`, to prevent the overflow or quantization error when processed in `np.uint8`. My usual way to read an image and make it gray-scale:

```python=
from skimage import io
from skimage import util
from skimage import color

img = io.imread('test.png')
img = util.img_as_float(img)
img = color.rgb2gray(img) # [H, W 3] -> [H, W]
```

### Showing Image

`Matplotlib` has [2 interfaces](https://medium.com/@kapil.mathur1987/matplotlib-an-introduction-to-its-object-oriented-interface-a318b1530aed). One is MATLAB style, one is object oriented. While most tutorial use former, I personally prefer latter. Following are some examples.

To show an image:
```python=
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()
```
The `imshow` function can show both `np.uint8` image and `np.float` image. It expects `np.uint8` having range `[0, 255]` and `np.float` image `[0.0, 1.0]`. When showing gray-scale image, `imshow` can have `cmap` ([colormap](https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html)) as argument. It specifies which color each value to mapping to. By default, `cmap` is `viridis`. If you want gray-scale output, `cmap` should be set as `gray`:

```python=
ax.imshow(img, cmap='gray')
```

To draw 2x3 plots in a figure:

```python=
fig, ax = plt.subplots(nrows=2, ncols=3) # ax is ndarray shaped [2, 3]
ax[0, 0].imshow(...)
ax[1, 0].imshow(...)
...
plt.show()
```

To overlap the image with other plots, say some 2d points:
```python=
r, c = np.nonzero(mask)
print(r.shape) # [#points, ]
print(c.shape) # [#points, ]

fig, ax = plt.subplots()
ax.imshow(...)
ax.plot(c, r, 'r.', markersize=3) # or ax.scatter(c, r, ...)
plt.show()
```