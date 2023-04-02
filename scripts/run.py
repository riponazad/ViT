import torch
import torchvision.transforms as transforms
from PIL import Image
from vit import ViT
import sys
import utils



if __name__ == '__main__':
    # Load the image and resize to the required size
    img = Image.open(sys.argv[1]).resize((224, 224))
    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply the transformation to the image
    img = transform(img)

    # Add a batch dimension to the image
    img = img.unsqueeze(0)
    #utils.show_image(img[0])

    # Instantiate the ViT model
    model = ViT(image_size=224, patch_size=16, num_classes=10, embed_dim=768, num_heads=12, num_layers=12)

    # Load the pretrained weights
    #model.load_state_dict(torch.load('ViT_pretrained.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Run the image through the model and get the predicted class
    with torch.no_grad():
        outputs = model(img)
        print(outputs.shape)
        #utils.show_image_patches(outputs)
        #_, predicted = torch.max(outputs, 1)

    # Print the predicted class
    #print('Predicted class:', predicted.item())
    #img.show()