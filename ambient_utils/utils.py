import torch
import numpy as np
import PIL
import PIL.Image
import torchvision.transforms as transforms
import math
import os
from typing import Any

def get_rel_methods(obj, keyword):
    """Returns all methods/properties of obj that contain keyword in their name.
    """
    return [attr for attr in dir(obj) if keyword in attr]

def broadcast_batch_tensor(batch_tensor):
    """ Takes a tensor of potential shape (batch_size) and returns a tensor of shape (batch_size, 1, 1, 1).
    """
    return batch_tensor.view(-1, 1, 1, 1)


def ambient_sqrt(x):
    """
        Computes the square root of x if x is positive. If not, it returns 1.
    """
    return torch.where(x < 0, torch.ones_like(x), x.sqrt())


def load_image(image_obj, device='cuda', resolution=None):
    if type(image_obj) == str:
        pil_image = PIL.Image.open(image_obj)
    elif type(image_obj) == PIL.Image.Image:
        pil_image = image_obj
    else:
        raise ValueError(f"Unrecognized image type: {type(image_obj)}")
    if resolution is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    tensor_image = transform(pil_image)
    return torch.unsqueeze(tensor_image, 0).to(device)

def save_image(images, image_path, save_wandb=False, down_factor=None, wandb_down_factor=None, 
               image_type="RGB"):
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if image_np.shape[2] == 1:
        pil_image = PIL.Image.fromarray(image_np[:, :, 0], 'L')
    else:
        pil_image = PIL.Image.fromarray(image_np, image_type)
    if down_factor is not None:
        pil_image = pil_image.resize((pil_image.size[0] // down_factor, pil_image.size[1] // down_factor))
    
    pil_image.save(image_path)
    
    if save_wandb:
        import wandb

    if save_wandb and wandb.run is not None:
        if wandb_down_factor is not None:
            # resize for speed
            pil_image = pil_image.resize((pil_image.size[0] // wandb_down_factor, pil_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(pil_image)})


def find_closest_factors(number):
    sqrt_number = int(math.sqrt(number))
    
    n = sqrt_number
    m = number // n
    
    while n * m != number:
        n += 1
        m = number // n

    return m, n


def save_images(images, image_path, num_rows=None, num_cols=None, save_wandb=False, down_factor=None, wandb_down_factor=None):
    if num_rows is None and num_cols is None:
        num_rows = int(np.sqrt(images.shape[0]))    
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    elif num_rows is None and num_cols is not None:
        num_rows = int(np.ceil(images.shape[0] / num_cols))
    elif num_rows is not None and num_cols is None:
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    
    if num_rows * num_cols != images.shape[0]:
        num_rows, num_cols = find_closest_factors(images.shape[0])
    
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_size = images.shape[-2]
    grid_image = PIL.Image.new('RGB', (num_cols * image_size, num_rows * image_size))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            img = PIL.Image.fromarray(image_np[index])
            grid_image.paste(img, (j * image_size, i * image_size))

    if down_factor is not None:
        grid_image = grid_image.resize((grid_image.size[0] // down_factor, grid_image.size[1] // down_factor))
    
    grid_image.save(image_path)

    if save_wandb:
        import wandb

    if save_wandb and wandb.run is not None:
        if wandb_down_factor is not None:
            # resize for speed
            grid_image = grid_image.resize((grid_image.size[0] // wandb_down_factor, grid_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(grid_image)})



def tile_image(batch_image, n, m=None):
    if m is None:
        m = n
    assert n * m == batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, m, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)  # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, m * width)
    return batch_image


_dnnlib_cache_dir = None


def stylize_plots():
    import seaborn as sns
    sns.set(style="whitegrid")


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def expand_vars(args):
    if not isinstance(args, dict):
        args = EasyDict(vars(args))
    else:
        args = EasyDict(args)
    for key, value in args.items():
        if isinstance(value, str) and "$" in value:
            args[key] = os.path.expandvars(value)
    return args  

def set_seed(seed=42):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_file(filename):
    if not filename.startswith('s3://'):
        return os.path.isfile(filename)
    else:
        import s3fs
        s3 = s3fs.S3FileSystem(anon=False)
        return s3.isfile(filename)


def pad_image(image, mode='reflect', height_patch=14, width_patch=14):
    # Get the input image shape
    batch, channels, height, width = image.shape
    
    # Calculate the padding required for height and width
    padding_height = (height_patch - (height % height_patch)) % height_patch
    padding_width = (width_patch - (width % width_patch)) % width_patch
    
    # Determine the padding on each side
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding
    left_padding = padding_width // 2
    right_padding = padding_width - left_padding
    
    # Apply the padding to the image tensor
    padding = (left_padding, right_padding, top_padding, bottom_padding)
    padded_image = torch.nn.functional.pad(image, padding, mode=mode)

    return padded_image


def ensure_tensor(func):
    def wrapper(*args, **kwargs):
        first_arg = args[0]
        if isinstance(first_arg, np.ndarray):
            args = (torch.tensor(first_arg),) + args[1:]
        func_result = func(*args, **kwargs)
        if isinstance(first_arg, np.ndarray):
            return func_result.cpu().numpy()
        return func_result
    return wrapper

def ensure_dimensions(func):
    def wrapper(*args, **kwargs):
        first_arg = args[0]
        if first_arg.ndim == 3:
            args = (first_arg.unsqueeze(0),) + args[1:]
        func_result = func(*args, **kwargs)
        if first_arg.ndim == 3 and func_result.ndim == 4:
            return func_result.squeeze(0)
        return func_result
    return wrapper


def bucketize(values, num_buckets):
    """
        Buckets the values into num_buckets buckets.
        Args:
            values: (batch_size,)
            num_buckets: int
        Returns:
            bucket_indices: (batch_size,)
    """
    # find min and max of values, create num_buckets buckets uniformly between them and split the values into num_buckets arrays
    min_value = values.min()
    max_value = values.max()

    # find a value every (max_value - min_value) / num_buckets apart
    bucket_values = torch.linspace(min_value, max_value, num_buckets + 1, device=values.device)
    bucket_indices = torch.bucketize(values, bucket_values)
    return bucket_indices


def image_to_numpy(image):
    return (image * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def image_from_numpy(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 127.5 - 1