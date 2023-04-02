from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms.functional as TF


def create_canvas(images, rows, cols, canvas_width=800, canvas_height=800, padding=10):
    canvas = Image.new(mode='RGB', size=(canvas_width, canvas_height), color='white')
    canvas_draw = ImageDraw.Draw(canvas)
    
    image_width = (canvas_width - padding * (cols + 1)) // cols
    image_height = (canvas_height - padding * (rows + 1)) // rows
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        img_resized = img.resize((image_width, image_height), resample=Image.BILINEAR)
        x = padding + col * (image_width + padding)
        y = padding + row * (image_height + padding)
        canvas.paste(img_resized, (x, y))
        canvas_draw.rectangle([(x, y), (x + image_width, y + image_height)], outline='black', width=3)
        
    canvas.show()
    return canvas

def show_image_patches(x):
    """
    Function to show image patches of ViT on a canvas.
    
    Args:
    - x: Tensor of shape (batch_size, channels, num_patches, patch_dim)
    """
    batch_size, num_patches, channels, patch_size, patch_size = x.shape
    grid_size = int(num_patches ** 0.5)
    
    patchs = [TF.to_pil_image(patch) for patch in x[0]]
    create_canvas(images=patchs, rows=grid_size, cols=grid_size)


def show_image(img_tensor):
    img_pil = TF.to_pil_image(img_tensor)
    img_pil.show()
