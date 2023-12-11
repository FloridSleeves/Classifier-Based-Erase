# Classifier-Based-Erase
This is the repo containing code for erasing a target visual concept from a pretrained stable diffusion model with the aid of a classifier for the target concept.

## Erase Concept
We currently support the following concepts: `['Albert Bierstadt', 'Albrecht Durer', 'Alfred Sisley', 'Amedeo Modigliani', 'car'].`
To erase a certain concept, one can run the command
```
python train.py --concept_prompt="Albert Bierstadt"
# checkpoint/car/erase.pt is the finetuned checkpoint
```
One can also manipulate the hyperparameters in the erasing process
```
python train.py --concept_prompt="Albert Bierstadt" --iterations=100 --lr=0.03 --nsteps=50
```
* `concept_prompt` specifies the concept to be erased. It is also used as the training prompt for erasure.
* `iterations`: number of fine-tuning iterations.
* `lr`: learning rate of the SGD optimizer.
* `nsteps`: maximum number of stpes used in sampling latent variables from the diffuse model.

## Test model
To test whether the target visual concept has been erased from a model,
one can run the command
```
python test.py --prompt="Albert Bierstadt Paint" --model_path="./checkpoint/Albert_Bierstadt/erase.pt" --concept="Albert Bierstadt" --n_imgs=10
```
The provided prompt is the testing prompt used to generate images from the model specified in the model_path. The generated images will be saved under an image folder.

## File Organization
* `detector` folder contains the data for the pretrained classifier. "artist" folder contains the pretrained classifier for the artists: 0. Albert Bierstadt 1. Albrecht Durer 2. Alfred Sisley 3. Amedeo Modigliani. "car" folder contains the class information of the pretrained resnet50 network.
* `classifier.py` contains our implementation of the soft detectors for the supported concepts.
In particular, given latent variables from a diffusion model as input, it estimates the log-likelihood that the associated image contains the target visual concept and computes the gradient with respect to the latent variables. The class "ArtModel" is used for detecting artistict styles and "CarModel" is used for detecting the presence of cars.
* `StableDiffusser.py` abstracts out the steps for performing sampling with a stable diffusion model.
* `finetuning.py` defines a context for switching between finetuning model weights and ground model weights.
* `train.py` contains the training script for erasing a target concept from the pretrained stable diffusion model from [here](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation)
* `test.py` contains the script for testing whether a target concept has been successfully erased.
