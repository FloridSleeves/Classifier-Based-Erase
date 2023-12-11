import json
import fire, os, glob
# Load imagenet label
labels = []
with open("./detector/car/resnet_label.json") as f:
    labels = json.load(f)
car_related_labels = ['taxicab', 'race car', 'passenger car', 'convertible','station wagon', 'pickup truck', 'grille', 'sports car', 'car wheel','minivan', 'jeep','tow truck', 'freight car', 'tow truck']
car_indices = [labels.index(label) for label in car_related_labels]    

import torch
from torchvision import transforms
from PIL import Image
# Load Resnet50 weights
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F


# Load pre-trained ResNet-50 model
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet50.eval().to("cuda:0")  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_car(imgA):
    return torch.argmax(resnet50(transform(imgA).unsqueeze(0).cuda().to("cuda:0"))) in car_indices

def create_model(self, num_artists):
    import torchvision
    # transfer learning on top of ResNet (only replacing final FC layer)
    model_conv = torchvision.models.resnet18(pretrained=True)
    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_artists)
    # load the pre-trained weights
    model_conv.load_state_dict(torch.load('./detector/artist/artist_ckp/state_dict.dat.von_gogh'))
    return model_conv
model = create_model(5)

def is_art(imgA, artist_id):
    return torch.argmax(model(transform(imgA).unsqueeze(0).cuda().to("cuda:0"))) == artist_id

from StableDiffuser import StableDiffuser
from finetuning import FineTunedModel
import torch
from tqdm import tqdm
from train import train

from StableDiffuser import StableDiffuser
from finetuning import FineTunedModel
import torch
from tqdm import tqdm



import torch.nn as nn


supported_concepts = ['Albert Bierstadt', 'Albrecht Durer', 'Alfred Sisley', 'Amedeo Modigliani', 'car']

def test(prompt, model_path="./our_models/car.pt", concept="car", n_imgs=10):
    if concept not in supported_concepts:
        raise ValueError(f"Concept prompt {concept} not supported. Supported concepts: {supported_concepts}")
    target_id = supported_concepts.index(concept)

    diffuser = StableDiffuser(scheduler='DDIM').to('cuda:0').eval().half()

    del diffuser.safety_checker
    finetuner = FineTunedModel.from_checkpoint(diffuser, model_path).eval().half()
    with finetuner:
        images = diffuser(
            prompt,
            n_steps=50,
            n_imgs=n_imgs,
            guidance_scale=3,
            noise_level=0,
        )
    record = []    
    for index, image in enumerate(images):
        # Detect
        if concept == "car":
            if is_car(image[0]):
                record.append(index)
        else:
            if is_art(image[0], target_id):
                record.append(index)
        image[0].save(f"image/sample{index}.model.png")    
    print(f"Under prompt:{prompt}, the percentage of image of input model containing car is {len(record)/n_imgs}")        

    images = diffuser(
        prompt,
        n_steps=50,
        n_imgs=n_imgs,
        guidance_scale=3,
        noise_level=0,
    )        
    
    record = []
    for index, image in enumerate(images):
        # Detect
        if concept == "car":
            if is_car(image[0]):
                record.append(index)
        else:
            if is_art(image[0], target_id):
                record.append(index)
        # Save the image
        image[0].save(f"image/sample{index}.ground.png")
    print(f"Under prompt:{prompt}, the percentage of image of ground model containing car is {len(record)/n_imgs}")

if __name__ == '__main__':    
    fire.Fire(test)