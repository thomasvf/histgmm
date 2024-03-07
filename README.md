# Histogram GMM
A Gaussian Mixture Model implementation that works with a histogram instead of using raw data points.

This implementation is useful for fitting data that has the form of a sum of gaussian curves, but it can also be used as a typical GMM implementation by computing a histogram of data points.

## Installation
As of now, this is not on PyPI, so the easiest way to install it is by cloning this repo and then installing it with pip:
```shell
git clone https://github.com/thomasvf/histgmm.git
cd histgmm
pip install .
```

## Usage

The example below shows how `HistogramGMM` can be used to fit the GMM to a sum of gaussians. 
It's also available as a Jupyter notebook [here](examples/basic_example.ipynb).

```python
import numpy as np

from histgmm import HistogramGMM
from histgmm.utils import gaussian_1d

# create data for the example
x = np.arange(0, 100, 1)
h = gaussian_1d(x, A=0.2, mu=10, var=9) + \
    gaussian_1d(x, A=1, mu=35, var=16) + \
    gaussian_1d(x, A=0.7, mu=46, var=25) + \
    gaussian_1d(x, A=1, mu=65, var=25)


# fit using 4 components
model = HistogramGMM(n_components=4, n_dimensions=1)
model.fit(x.reshape(-1, 1), h)
```

The fitted gaussians can be plotted over the original data using the function `plot_1d_gaussian_fit`, as shown below.

```python
import matplotlib.pyplot as plt
from histgmm.visualization import plot_1d_gaussian_fit

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(x, h, marker='.')
ax[0].set_title('Original Curve')

plot_1d_gaussian_fit(gmm=model, X=x.reshape(-1, 1), h=h, ax=ax[1])
ax[1].set_title('Fitted and Scaled Gaussians')

plt.show()
```
<p align="center">
  <img src="./examples/images/1d_gaussian_mixture_fit.png" alt="1d gaussian mixture" width="100%">
</p>


## Model Formulation
This code adapts the EM for Gaussian Mixture Models found in Bishop's 2006 book to work with histograms.

Essentially, we work with bins instead of examples, and each bin is associated with a certain number of examples indicated by the histogram.
That is, $\bold{x}_b$ gives the $M-$dimensional position of bin $b$, and $h_b$ is the number of examples in that bin (that is, that have this same position).
We then only need to change the summations so that they emulate the iteration over all the examples in a bin.
Since all bins have the same position, this simply means multiplying the summed term by the appropriate value in $h_b$.

### E-Step
The E-step evaluate the responsabilities $r_{nk}$ of each cluster $k=1, ..., K$ and point $\bold{x}_n$.
$r_{nk}$ is also equivalent to $p(k|\bold{x}_n)$, that is, the probability of a certain cluster $k$ given example $x_n$.
In our case, we deal with the positions of each bin $b$.
Since all examples in a bin have the same position, we only need to compute the responsabilities $r_{bk}$ for each bin position $\bold{x}_b$.

The E-step becomes:
$$
r_{bk} = \frac{\pi_k \mathcal{N}(\bold{x}_b | \boldsymbol{\mu}_k, \bold{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\bold{x}_b | \boldsymbol{\mu}_k, \bold{\Sigma}_k)}.
$$

Note that we have simply changed the index from $n$ to $b$.

### M-Step
We adapt the M-step to emulate the iteration over all the examples in all bins.
Let $h_b$ be the number of examples in bin $b$ for $b=1, ..., B$.

The total number of examples and the effective number of examples in a cluster become, respectively:
$$
N = \sum_{b=1}^{B} h_b
$$
and 
$$
N_k = \sum_{b=1}^{B} h_b r_{bk}.
$$

The estimations of the gaussian parameters are similarly modified:
$$
\boldsymbol{\mu}_k^{new} = \frac{1}{N_k} \sum_{b=1}^B h_b\ r_{bk}\ \bold{x}_b\
$$

$$
\bold{\Sigma}_k^{new} = \frac{1}{N_k} \sum_{b=1}^B h_b\ r_{bk}\ 
(\bold{x}_b - \boldsymbol{\mu}_k^{new}) (\bold{x}_b - \boldsymbol{\mu}_k^{new})^\mathsf{T}
$$

$$
\pi_k^{new} = \frac{N_k}{N}
$$

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
