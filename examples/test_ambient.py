import ambient_utils as ambient
import numpy as np
import math
import torch

def main():
    print("Creating a dataset from an image folder...")



    dataset = ambient.ImageFolderDataset(
        path="image_folder",
        only_positive=False, # images will be in [-1, 1]
    )  # you can also use another dataser that inherits from the torch.utils.data.Dataset class

    print("Dataset created.")
    print("Let's add some annotations to the dataset...")
    for i in range(len(dataset)):
        # let's add some annotations now. This decides which diffusion times are allowed.
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

    print("Annotations added.")

    print("Will create an infinite Ambient sampler from the dataset...")

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

    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=4)
    print("Dataloader created.")

    # get a batch
    batch = next(iter(dataloader))

    ambient.save_images(batch['image'], "test_batch.jpg", num_rows=1)
    print("Saved images to test_batch.jpg")

    print("Will now do an example training loop...")
    print("First, we extract the trust level.")
    sigma_tn = torch.tensor([sampler.sampled_sigmas[i.item()]['sigma_min'] for i in batch['idx']])
    print("Next, we extract the noise level that we want to do the training for.")
    sigma_t = torch.tensor([sampler.sampled_sigmas[i.item()]['sigma'] for i in batch['idx']])
    sigma_tn = torch.where(sigma_tn > sigma_t, torch.zeros_like(sigma_tn), sigma_tn) # make sure we ground truth version we have for the sample is at less noise.
    print("Let's noise the image to the trust level.")
    image_tn = batch['image'] + batch.get('noise', torch.zeros_like(batch['image'])) * sigma_tn[:, None, None, None] # corrupt the image to the noise level that we can trust them.
    print("Let's now further noise the image to the level we want to do the training for.")
    image_t = image_tn + torch.randn_like(batch['image']) * torch.sqrt(sigma_t[:, None, None, None] ** 2 - sigma_tn[:, None, None, None] ** 2) # add noise to the image to the noise level that we want to do the training for.

    print("Let's save the original image, the image in the trust level and the image in the training level to test_batch_corrupted.jpg...")
    ambient.save_images(torch.cat([batch['image'], image_tn, image_t], dim=0), "test_batch_corrupted.jpg", num_rows=3) # row 1: original image, row 2: corrupted image at trust level, row 3: corrupted image at the noise level that we want to do the training for.
    print("Saved to test_batch_corrupted.jpg...")




    





        









if __name__ == "__main__":
    main()