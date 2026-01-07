# robotics_transformer/data/transforms.py
from torchvision.transforms import v2 as T

# Standard ImageNet statistics for pre-trained models (EfficientNet, ResNet, etc.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_image_transform(image_size: int):
    """
    Prepares raw numpy images for the EfficientNet-B3 vision backbone.
    ToImage -> ToDtype -> Resize -> Normalize
    """
    return T.Compose([
        T.ToImage(), # Converts to tensor without scaling
        T.ToDtype(torch.float32, scale=True), # Scales pixels to [0, 1]
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])