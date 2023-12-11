import json
import torch
# Load Resnet50 weights
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
from diffusers import AutoencoderKL

class ImageConverter(nn.Module):
    def __init__(self):
        super(ImageConverter, self).__init__()
        # mean/std required by resnet
        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])
        self.val_transform = T.Compose([T.Resize((256, 256)),T.CenterCrop(224),T.Normalize(mean=mean_resnet, std=std_resnet)])

    def forward(self, input):
        MAX = torch.max(input).detach()
        MIN = torch.min(input).detach()
        input = (input - MIN) / (MAX - MIN)
        return self.val_transform(input)

class ArtModel(nn.Module):

    def __init__(self, artist_id):
        super(ArtModel, self).__init__()
        self.rgb = ImageConverter()
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda:1")
        self.vae.eval()
        self.classifier = self.create_model(5)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.artist_id = artist_id

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

    def compute_prob_from_output(self, output):
        print("Target Class: ", torch.argmax(output[0]))
        return output[0][self.artist_id] - torch.logsumexp(output[0], 0)

    def forward(self, x):
        x = self.vae.decode(1 / self.vae.config.scaling_factor * x).sample
        x = self.rgb(x)
        x = self.classifier(x)
        return self.compute_prob_from_output(x)

class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        self.rgb = ImageConverter()
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda:1")
        self.vae.eval()
        self.classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.classifier.eval().to("cuda:0")
        for param in self.classifier.parameters():
            param.requires_grad = False   

    def compute_prob_from_output(self, output):
        labels = []
        with open("./detector/car/resnet_label.json") as f:
            labels = json.load(f)
        car_related_labels = ['taxicab', 'race car', 'passenger car', 'convertible','station wagon', 'pickup truck', 'grille', 'sports car', 'car wheel','minivan', 'jeep','tow truck', 'freight car', 'tow truck']
        car_indices = [labels.index(label) for label in car_related_labels]    
        car_weight = output[0][car_indices]
        return torch.logsumexp(car_weight, 0) - torch.logsumexp(output[0], 0)

    def forward(self, x):
        x = self.vae.decode(1 / self.vae.config.scaling_factor * x).sample
        x = self.rgb(x)
        x = self.classifier(x)
        return self.compute_prob_from_output(x)