from PIL import Image
from IPython.display import display
import torch as th
import torch
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from glide_text2im.soft_embedding import SoftEmbedding
import torch.nn as nn
import numpy as np


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda:6')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '10' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

for param in model.parameters():
    param.requires_grad_(False)

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = '2' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

for param in model_up.parameters():
    param.requires_grad_(False)

def show_images(batch: th.Tensor, path):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # display(Image.fromarray(reshaped.numpy()))
    pil_image = Image.fromarray(reshaped.numpy())
    pil_image.save(path)


def load_image_as_tensor(path):
    pil_image = Image.open(path).convert('RGB')
    numpy_image = np.array(pil_image)
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
    tensor_image = torch.from_numpy(numpy_image).float()
    tensor_image = (tensor_image / 127.5) - 1
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)


######################## initialize the label ################

label = load_image_as_tensor('/home/changsheng/glide-text2im/output_image_bomb.png')
label = label.to(device)


######################### initialize the tokens ################
# Sampling parameters
prompt = "an oil painting of a corgi"
batch_size = 1
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997
tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Create the classifier-free guidance tokens (empty)
full_batch_size = batch_size * 2
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

tokens=th.tensor(
        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
    )

# tokens size: torch.Size([2, 128, 512])

################### initialize the embedding ###############

n_tokens = 5
initialize_from_vocab = True
s_wte = SoftEmbedding(nn.Embedding(50257, 512), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=initialize_from_vocab)
s_wte.to(device)

# embeding size : torch.Size([2, 128, 512])
####################### Train process #############

from torch.optim import Adam
import torch.nn.functional as F

# Define the loss function (L2 distance)
def l2_loss(tensor1, tensor2):
    """Calculate the L2 loss between two tensors."""
    return F.mse_loss(tensor1, tensor2)


# self_emb = torch.rand(2, 768) * 2 - 1  
self_emb = torch.load('/home/changsheng/glide-text2im/ck/cogi_emb.pt')
print(self_emb.shape)
self_emb = self_emb.to(device)  
self_emb.requires_grad_(True)  

# Initialize the optimizer
# optimizer = Adam([self_emb.requires_grad_()], lr=1)
optimizer = Adam(s_wte.parameters(), lr=0.001)
# Number of optimization steps
n_steps = 1000

embeding = s_wte(tokens)

for step in range(n_steps):
    optimizer.zero_grad()  # Reset gradients to zero
    # ------- model start ----
    
    prompt = "an oil painting of a corgi"
    batch_size = 1
    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        )
    
    embeding = s_wte(tokens)
    print("embeding:", embeding.requires_grad)
    model_kwargs = dict(
        tokens=tokens ,
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
        # Randomly initialize an embedding with values in the range 0 to 0.5
        # with the shape [2, 128, 512]
        self_init = embeding,
        self_emb = self_emb,
    )
    
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()
    path = 'output_image_64.png'
    show_images(samples,path)
    
    print("samples:", samples.requires_grad)
    
    ##### finish base model --> 64-256 up model ####
    
    prompt = "generate a image that"
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )
    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
        self_init = (embeding)[:1],
        self_emb = self_emb[:1],
    )
    print("embeding:", embeding.requires_grad)
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()
    
    print("up_samples:", up_samples.requires_grad)
    
    path = 'output_image_256.png'
    show_images(up_samples,path)
    
    # up_samples size :  torch.Size([1, 3, 256, 256])

    
    print("up_samples:", up_samples.requires_grad)
    
    # ----- model end ----
    loss = l2_loss(label, up_samples)
    
    # Backpropagation
    loss.backward()
    # Update parameters
    optimizer.step()
    # Print loss every 100 steps
    # if step % 100 == 0:
    print(f"Step {step}, Loss: {loss.item()}")
    break

# Save the optimized embedding
torch.save(s_wte.state_dict(), 'ck/optimized_s_wte.pt')






