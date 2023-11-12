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
from torch.optim import Adam,SGD
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from torchvision import transforms

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda:6')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '5' # use 100 diffusion steps for fast sampling
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

# Define the loss function (L2 distance)
def l2_loss(tensor1, tensor2):
    """Calculate the L2 loss between two tensors."""
    return F.mse_loss(tensor1, tensor2)

def prompt_token(options,device,model):
    prompt = "an oil painting of a corgi"
    batch_size = 1
    guidance_scale = 3.0
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
    
    masks = th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        )
    
    return tokens,masks,full_batch_size 


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(True)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

class config:
    mm_vision_tower='openai/clip-vit-large-patch14'
    # mm_vision_tower='openai/clip-vit-large-patch14-336'
    # openai/clip-vit-large-patch14-336
    mm_vision_select_layer=-2
    mm_vision_select_feature='patch'
    
from transformers import CLIPProcessor, CLIPModel

clip = build_vision_tower(config)
clipProcessor = CLIPProcessor.from_pretrained(config.mm_vision_tower)

def get_parameters_snapshot(model):
    return {name: param.clone().detach() for name, param in model.named_parameters()}

def compare_parameters(snapshot1, snapshot2):
    for name in snapshot1:
        if not torch.equal(snapshot1[name], snapshot2[name]):
            return True
    return False




######################## initialize the label ################

label = load_image_as_tensor('/home/changsheng/glide-text2im/output_image_bomb.png')
label = label.to(device)


######################### initialize the tokens ################
# Sampling parameters

tokens,masks,full_batch_size  = prompt_token(options,device,model)

# tokens size: torch.Size([2, 128, 512])

################### initialize the embedding ###############


# embeding size : torch.Size([2, 128, 512])
####################### Train process #############

# self_emb = torch.rand(2, 128, 512) * 2 - 1 

self_emb = torch.load('/home/changsheng/glide-text2im/ck/xf_in_tensor.pt')
self_emb = self_emb.to(device)
self_emb.requires_grad_(True)


optimizer = SGD([self_emb.requires_grad_(True)], lr=0.01)
# Number of optimization steps
n_steps = 10000
criterion = nn.MSELoss()


param_snapshots = []
for step in range(n_steps):
    optimizer.zero_grad()  # Reset gradients to zero
    # ------- model start ----
    
    guidance_scale = 3.0
    upsample_temp = 0.997
    # print(self_emb[:1,5:10,:])
    # print("embeding:", self_emb.requires_grad)
    batch_size = 1
    model_kwargs = dict(
        tokens=tokens ,
        mask=masks,
        # Randomly initialize an embedding with values in the range 0 to 0.5
        # with the shape [2, 128, 512]
        self_init = self_emb,
        self_emb = None,
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
    
    # print("samples:", samples.requires_grad)
    
    ##### finish base model --> 64-256 up model ####
    
    prompt = "an oil painting of a corgi"
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
        self_init = (self_emb)[:1],
        self_emb = None,
    )
    # print("embeding:", embeding.requires_grad)
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
    
    # print("up_samples:", up_samples.requires_grad)
    
    path = 'output_image_256.png'
    show_images(up_samples,path)
    
    # up_samples size :  torch.Size([1, 3, 256, 256])
    # print("up_samples:", up_samples.requires_grad)
    
    ################ Clip Model ##############
    # torch.Size([1, 3, 224, 224])
    
    up_samples_224 = F.interpolate(up_samples, size=(224, 224))
    label_224 = F.interpolate(label, size=(224, 224))

    h_adv = clip(up_samples_224)
    h_harm = clip(label_224)
    loss = criterion(h_adv, h_harm)
    
    # ----- model end ----
    # loss = l2_loss(label, up_samples)
    # Backpropagation
    loss.backward()
    # Update parameters
    optimizer.step()
    # param_snapshots.append(get_parameters_snapshot(model))
    # Print loss every 100 steps
    # if step % 100 == 0:
    print(f"Step {step}, Loss: {loss.item()}")
    
# Save the optimized embedding
# torch.save(s_wte.state_dict(), 'ck/optimized_s_wte.pt')
'''
for i in range(1, len(param_snapshots)):
    if compare_parameters(param_snapshots[i-1], param_snapshots[i]):
        print(f"Epoch {i}: Parameters changed.")
    else:
        print(f"Epoch {i}: No change in parameters.")

'''



