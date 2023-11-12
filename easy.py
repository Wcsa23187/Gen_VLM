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

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
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
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

for param in model_up.parameters():
    param.requires_grad_(False)

# Sampling parameters
prompt = "generate a image that"
batch_size = 1
guidance_scale = 3.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
tokens = model.tokenizer.encode(prompt)

print(tokens)

tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Init an embedding 
xf_in = torch.load('ck/xf_in_tensor.pt')
assert xf_in.shape == torch.Size([2, 128, 512]), "The tensor does not have the expected shape."

print(xf_in)

# Split the tensor into two halves
half_size = xf_in.shape[0] // 2
tokens_emb = xf_in[:half_size]
uncond_tokens_emb = xf_in[half_size:]

# Save each half to a file
torch.save(tokens_emb, 'ck/tokens_emb.pt')
torch.save(uncond_tokens_emb, 'ck/uncond_tokens_emb.pt')

n_tokens = 20
initialize_from_vocab = True

s_wte = SoftEmbedding(nn.Embedding(50257, 512), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=initialize_from_vocab)
s_wte.to(device)
# Create the classifier-free guidance tokens (empty)
full_batch_size = batch_size * 2
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

tokens=th.tensor(
        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
    )

embeding = s_wte(tokens)

# xf_in
from torch.optim import Adam
import torch.nn.functional as F

# Define the loss function (L2 distance)
def l2_loss(tensor1, tensor2):
    """Calculate the L2 loss between two tensors."""
    return F.mse_loss(tensor1, tensor2)

# Initialize the optimizer
optimizer = Adam(s_wte.parameters(), lr=0.001)

# Number of optimization steps
n_steps = 1000


xf_in = xf_in.to(torch.float32)
print(xf_in.dtype)
# Optimization loop
for step in range(n_steps):
    optimizer.zero_grad()  # Reset gradients to zero
    embedding = s_wte(tokens)  # Get the current embeddings
    print(embedding)
    print(embedding.shape)
    # print(embedding.dtype)
    # Calculate loss
    print("embedding:", embedding.requires_grad)
    print("xf_in:", xf_in.requires_grad)
    loss = l2_loss(embedding, xf_in)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# Save the optimized embedding
torch.save(s_wte.state_dict(), 'ck/optimized_s_wte.pt')


