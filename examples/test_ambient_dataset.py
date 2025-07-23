import ambient_utils as ambient
import numpy as np
import math
import torch

def main():
    print("Creating a dataset from an image folder...")

    dataset = ambient.ImageFolderDataset(
        path="image_folder",
        only_positive=False, # images will be in [-1, 1]
    )

    print("Dataset created.")
    print("Let's add some annotations to the dataset...")
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        filename = sample["filename"]
        noise = sample["noise"]

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
        infinite=False,
    )

    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=10)
    print("Dataloader created.")

    # get the first batch
    first_batch = next(iter(dataloader))

    ambient.save_images(first_batch['image'], "test_batch.jpg")
    print("Saved images to test_batch.jpg")

    for _ in dataloader:
        print(".")


        









if __name__ == "__main__":
    main()