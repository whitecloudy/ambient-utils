<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/giannisdaras/ambient-utils/blob/main/examples/example_image.jpg?raw=true" width="60%" alt="Ambient Omni Logo" />
</div>

## Package installation 👨‍💻

You can install this package by simply running:

```bash
pip install ambient-utils
```

If you plan to use all the functionalities of the library, you can use:
```bash
pip install ambient-utils[all]
```
but be aware that your installation will take a bit longer. For most use cases, just `pip install ambient-utils` should suffice.


## What is this about 🤨?

This repository hosts useful functions for training diffusion models (or flow matching models) in settings with limited access to high-quality data.
This repository provides the implementation of ideas around leveraging imperfect data sources, including low-quality, corrupted, synthetic, and out-of-distribution samples, to improve generalization without degrading sample quality. This problem appears in numerous scientific and practical applications. 

The code in this repository has been used in the following papers:

* [Ambient Diffusion Omni: Training Good Models with Bad Data](https://arxiv.org/abs/2506.10038) (preprint)
* [Ambient Proteins: Training Diffusion Models on Low Quality Structures](https://www.biorxiv.org/content/10.1101/2025.07.03.663105v1) (preprint)
* [Does generation require memorization? Creative Diffusion Models using Ambient Diffusion](https://arxiv.org/abs/2502.21278) (ICML 2025)
* [How much is a noisy image worth? Data Scaling Laws for Ambient Diffusion](https://arxiv.org/abs/2411.02780) (ICLR 2025)
* [Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data](https://arxiv.org/abs/2404.10177) (ICML 2024) 
* [Ambient Diffusion: Learning Clean Distributions from Corrupted Data](https://arxiv.org/abs/2305.19256) (NeurIPS 2023)
* [Consistent diffusion models: Mitigating sampling drift by learning to be consistent](https://arxiv.org/abs/2302.09057) (NeurIPS 2023)
* [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data](https://arxiv.org/abs/2403.08728) (ICLR 2025)

Beyond the functionality directly related to training diffusion models with corrupted data, this repository also provides a wide range of functions that can be useful for day-to-day deep learning projects. 


## What's a good place to start learning 📖?

If you are not familiar with the Ambient Diffusion family of papers, probably the best place to start is our [Ambient Diffusion Omni](https://arxiv.org/abs/2506.10038) work, as it contains the most polished versions of the ideas that we developed over the years.

If you are too lazy to do so, you can check out [this blogpost](https://giannisdaras.github.io/publication/ambient_omni) instead. 

If you are too lazy to do that, here is a TLDR.

> In diffusion modeling, the goal is to train denoisers for different noise levels. High-quality points are very useful as they help you learn denoisers for all noise levels. Lower quality / synthetic / out-of-distribution data are still useful, but they can only help you learn a subset of the denoisers. 



## How to use this framework in my codebase? ✨


It is pretty straightforward to use ideas for learning with bad data, add and integrate them into your existing standard diffusion codebase.
A standalone example is provided in the [`examples/test_ambient.py`](https://github.com/giannisdaras/ambient-utils/blob/main/examples/test_ambient.py) file.
Give or take, you will need to do 4 things.

### 1. Prepare your data

As mentioned before, each sample in your data will only help you learn for a subset of the diffusion times. Typically, a sample can be used only under high noise (typically a low-quality sample) or under very low noise (typically high-quality but out-of-distribution sample). 
We use `sigma_min` and `sigma_max` to indicate the allowed times. In particular, a sample can be used for all times t: $\sigma_t > \sigma_{\mathrm{min}} \ \vee \sigma_t < \sigma_{\mathrm{max}}$. These parameters can be exposed per sample using heuristics or domain knowledge. If you want a more principled way for selecting this per sample, see the [Ambient Omni paper](https://arxiv.org/abs/2506.10038).

You need to modify your existing `torch.utils.data.Dataset` to expose these two arguments. Our code expects the `dataset.annotations` property to be set. This should be a dictionary that maps the dataset index (integer) to a tuple (sigma_min, sigma_max). If this dictionary is not set, our code assumes that all samples can be used at any diffusion time (default behavior in other codebases).

Here is some dummy code for setting the annotations:

```python
for i in range(len(dataset)):
    # We select sigma_min and sigma_max. Image is used if sigma_t > sigma_min or sigma_t < sigma_max.
    # Typically, you want to do this based on some measure of quality.
    # For this example, let's just do it randomly.
    sigma_min = np.random.uniform(0.0, 4.0)
    sigma_max = np.random.uniform(0.0, 0.2)

    # also, let's make sure we have at least one fully clean image.
    if i == 0:
        sigma_min = 0.0
        sigma_max = math.inf

    sample_annotation = (sigma_min, sigma_max)

    dataset.annotations[i] = sample_annotation
```

It is further recommended to store a fixed noise for each sample in the dataset. The reasons for this will become clear in a bit. This can either be done by literally storing the noise array or by generating on the fly the noise based on a fixed per-image seed, such as the dataset index. The [`ambient_utils.dataset.Dataset`](https://github.com/giannisdaras/ambient-utils/blob/main/ambient_utils/dataset.py#L55) class already takes care of this functionality for you. But if you are working with a different `torch.utils.data.Dataset`, you have to implement this yourself.


### 2. Use the `AmbientSampler`

As mentioned, each sample can only be used for a subset of the diffusion times. This means that the standard way of first sampling a datapoint and then diffusion times no longer works, as we may get inadmissible pairs. Instead, we need to change the order: first sample a noise level $\sigma_t$, and then select from the pool of samples that can be used in that time, i.e. choose a sample for which $\sigma_{min} < \sigma_t$ or $\sigma_{max} > \sigma_t$. To make this easier, we have provided the class [`AmbientSampler`](https://github.com/giannisdaras/ambient-utils/blob/main/ambient_utils/dataset.py#L245) that takes care of this for you. 

Here is a very easy example on how to use this: 

```python
def scheduler_fn():
    """Return a random sigma value for the diffusion process."""
    rnd_normal = np.random.normal(0, 1)
    sigma = np.exp(rnd_normal * 1.2 - 1.2) # schedule from EDM paper for the VE SDE.
    return sigma

sampler = ambient.AmbientSampler(
    dataset,
    scheduler_fn,
    shuffle=True,
    infinite=False,  # if you set this to True, the sampler will loop over the dataset indefinitely
)

dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=16)
print("Dataloader created.")
```

### 3. Diffuse your data

Typically, there is some forward process that noises our images to timestep $t$ and then there is a network that tries to do the denoising. The easiest forward process is the Variance Exploding (VE) one, which simply adds noise of variance $\sigma^2_t$ to the image. For example, in a typical pipeline, you would noise images by doing something like:

```python
# add noise to the image to the noise level that we want to do the training for.
image_t = image + torch.randn_like(image) * sigma_t[:, None, None, None] 
```

In our framework, we perform this corruption in two steps: first, we corrupt the original images at a noise level $t_n$, and then we further corrupt them to bring them to noise level $t$. If you want to understand why this is necessary, I recommend you read (any) of these papers:
[Ambient Omni](https://arxiv.org/abs/2506.10038),[Does generation require memorization?](https://arxiv.org/abs/2502.21278).

The TLDR is that we can't trust the original image, but we can trust a noisy version of it at some noise level $t_n$. In some sense, we replace the potentially "bad" original image with a noisy version of it that we treat as a clean image + noise. Different images have different noise levels that they can be trusted at, as determined by `dataset.annotations`.

In any case, you can easily do this corruption by using the following code:

```python
sigma_tn = torch.tensor([sampler.sampled_sigmas[i.item()]['sigma_min'] for i in batch['idx']])
sigma_t = torch.tensor([sampler.sampled_sigmas[i.item()]['sigma'] for i in batch['idx']])
sigma_tn = torch.where(sigma_tn > sigma_t, torch.zeros_like(sigma_tn), sigma_tn) # make sure we ground truth version we have for the sample is at less noise.
image_tn = batch['image'] + batch.get('noise', torch.zeros_like(batch['image'])) * sigma_tn[:, None, None, None] # corrupt the image to the noise level that we can trust them.
image_t = image_tn + torch.randn_like(batch['image']) * torch.sqrt(sigma_t[:, None, None, None] ** 2 - sigma_tn[:, None, None, None] ** 2) # add noise to the image to the noise level that we want to do the training for.
```



### 4. Changing the diffusion loss to the ambient loss

The final (optional) step is to change the loss. You can skip this step without sacrificing much performance. However, if you really want to get the most of our framework, you are advised to change the loss function. 

Roughly speaking, the change of the loss is needed because if your datapoint is low-quality, you shouldn't use it as a target for your denoising objective. Instead, it is better to use the "noisy version" of your datapoint, `image_tn`, as your target and cleverly manipulate the loss function so that you get the same minimizer as you would get by observing the high-quality point that is not available.

If you want to understand more about this idea, read the work [Consistent Diffusion Meets Tweedie](https://arxiv.org/abs/2404.10177) or the [Ambient Omni](https://arxiv.org/abs/2506.10038) paper. If you are coming from the Computational Imaging community, this might remind you of [Noisier2Noise](https://arxiv.org/abs/1910.11908) (for good reasons).



```python
image_pred = dummy_network_fn(image_t, sigma_t)        
# bring this to the trust level
image_tn_pred = ambient.from_x0_pred_to_xnature_pred_ve_to_ve(image_pred, image_t, sigma_t, sigma_tn)


# this weighting is from the EDM paper.
sigma_data = 0.5
edm_weight = (sigma_data ** 2 + sigma_t ** 2) / (sigma_t ** 2 * sigma_data ** 2)
# this weighting is due to the change of the loss.
ambient_factor = sigma_t ** 4 / ((sigma_t ** 2 - sigma_tn ** 2) ** 2)
ambient_weight = edm_weight * ambient_factor
# loss computation
loss = ambient_weight[:, None, None, None] * ((image_tn_pred - image_tn) ** 2)
```


## How is this repository structured? 📝

* `datasets`: provides the `AmbientSampler`, a `torch.utils.data.Sampler` that allows for sampling different datapoints differently according to the strength of the corruption. This module also provides several utilities for working with ImageFolder datasets and WebDatasets.
* `classifier`: related to the [Ambient Diffusion Omni](https://arxiv.org/abs/2506.10038) work. Useful for parallel predictions and annotations using a pre-trained noise-dependent classifier.
* `diffusers`: provides useful functions for interoperability with the diffusers library. Used in the [Consistent Diffusion Meets Tweedie](https://arxiv.org/abs/2404.10177) work. Not under active development, please raise an issue if something is broken.
* `dist`: provides several functions for distributed training. 
* `eval`: provides implementations for Inception score and FID computation.
* `loss`: provides implementations for loss functions that compute the conditional expectation of the clean distribution without having access to clean data. This idea is related to Noisier2Noise.
* `url`: utility functions for working with URLs.
* `noise`: commonly used synthetic corruptions on images.
* `utils`: several other utility functions.

## Maintainers 👥

**Authors:**
- Giannis Daras (Postdoctoral Researcher, MIT)
- Adrian Rodriguez-Munoz (Ph.D. Student, MIT)

**Correspondence:** gdaras [at] mit [dot] edu

## Acknowledgements 🙏

We thank the authors of the [EDM repository](https://github.com/NVlabs/edm) for providing usable code blocks that we use in this repo.

## Citation 📚

If you use this package in your research, please consider citing the Ambient Omni paper:

```bibtex
@article{daras2024ambient,
  title={Ambient Diffusion Omni: Training Good Models with Bad Data},
  author={Daras, Giannis and Rodriguez-Munoz, Adrian and Klivans, Adam and Torralba, Antonio and Daskalakis, Constantinos},
  journal={arXiv preprint arXiv:2506.10038},
  year={2024}
}
```