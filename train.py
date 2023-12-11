import torch
from StableDiffuser import StableDiffuser
from finetuning import FineTunedModel
import torch
from tqdm import tqdm
from torch.autograd import Variable
from diffusers import AutoencoderKL
import fire, os, glob
import classifier

supported_concepts = ['Albert Bierstadt', 'Albrecht Durer', 'Alfred Sisley', 'Amedeo Modigliani', 'car']

class ObjectDetector():
    def __init__(self, concept):
        self.dtype = torch.FloatTensor
        if (torch.cuda.is_available()):
            self.dtype = torch.cuda.FloatTensor
        # transfer learning on top of ResNet (only replacing final FC layer)
        self.device = "cuda:0"
        if concept == "car":
            self.model = classifier.CarModel() 
        else:
            artist_id = supported_concepts.index(concept)
            self.model = classifier.ArtModel(artist_id)
        self.model.to(self.device)

    def get_input_grad(self, x): 
        x_var = Variable(x.type(self.dtype).to(self.device), requires_grad=True)
        prob = self.model(x_var)
        prob.backward()
        return x_var.grad



def train(concept_prompt="Alfred Sisley", save_path="./checkpoint/", iterations=100, lr=0.03, nsteps=50):
    if concept_prompt not in supported_concepts:
        raise ValueError(f"Concept prompt {concept_prompt} not supported. Supported concepts: {supported_concepts}")
    save_path = os.path.join(save_path, concept_prompt.replace(" ", "_")) + "/"
    os.makedirs(save_path, exist_ok=True)
    modules = ".*attn2$"
    freeze_modules=[]
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda:0')
    diffuser.train()
    finetuner = FineTunedModel(diffuser, modules, frozen_modules=freeze_modules)
    params = list(finetuner.parameters())
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():
        positive_text_embeddings = diffuser.get_text_embeddings([concept_prompt],n_imgs=1)

    del diffuser.vae
    del diffuser.text_encoder
    del diffuser.tokenizer
    del diffuser.safety_checker

    torch.cuda.empty_cache()

    detector = ObjectDetector(concept_prompt)
    optimizer = torch.optim.SGD(params, lr=lr)

    # Classifier Graident Needs to be scaled to match the scale of epsilon
    if concept_prompt == 'car':
        gradient_scale = 70
    else:
        gradient_scale = 40

    for i in pbar:
        with torch.no_grad():
            diffuser.set_scheduler_timesteps(50)

            optimizer.zero_grad()
            diffuse_iter = torch.randint(40, nsteps-1, (1,)).item()
            # if concept_prompt == 'car':
            #     diffuse_iter = torch.randint(40, nsteps-1, (1,)).item()
            # else:
            #     diffuse_iter = torch.randint(46, nsteps-1, (1,)).item()
            latents = diffuser.get_initial_latents(1, 512, 1)
            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=diffuse_iter,
                    guidance_scale=3,
                    show_progress=False,
                )
            
            with finetuner:
                ref_latents = diffuser.predict_noise(diffuse_iter, latents_steps[0], positive_text_embeddings, guidance_scale=1)


        with finetuner:
            curr_latents = diffuser.predict_noise(diffuse_iter, latents_steps[0], positive_text_embeddings, guidance_scale=1)
        
        input_x = latents_steps[0]    
        detector_grad = detector.get_input_grad(input_x)        

        loss = criteria(curr_latents.float(), ref_latents.detach().float() + gradient_scale*(detector_grad))
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            # delete preivous checkpoint
            for fname in glob.glob(save_path + f'mid_checkpoint.pt.*'):
                os.remove(fname)
            torch.save(
                finetuner.state_dict(), 
                save_path + f'mid_checkpoint.pt.{i}'
            )

    torch.save(finetuner.state_dict(), save_path + f'erase.pt')
    torch.cuda.empty_cache()

if __name__ == '__main__':    
    fire.Fire(train)