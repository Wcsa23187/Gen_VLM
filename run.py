from PIL import Image
from IPython.display import display
import torch as th
import torch
import torch.nn as nn
import torch.optim as optim
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModelForSeq2SeqLM, get_peft_config
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import random
from PIL import Image
import requests

seed_value=0
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # display(Image.fromarray(reshaped.numpy()))
    pil_image = Image.fromarray(reshaped.numpy())
    pil_image.save('output_image_make_bomb.png')
        
def Model_load():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '2' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    #####################
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
    return model , model_up, options, diffusion, diffusion_up,options_up

def Model_64(tokens, mask):
    
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

    #####################
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
    
    
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    
    #####################

    
    #######################
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

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

    # Sample from the base model.
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

    # Show the output
    show_images(samples)
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    
    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
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
    print(up_samples.shape)
    # Show the output
    show_images(up_samples) 

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

def display(tensor,path):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    stddev = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    normalized_image = tensor
    restored_image = normalized_image * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    to_pil = transforms.ToPILImage()
    tensor = (restored_image * 255).byte()
    pil_image = to_pil(tensor[0])
    pil_image.save(path)

class config:
    mm_vision_tower='openai/clip-vit-large-patch14'
    # mm_vision_tower='openai/clip-vit-large-patch14-336'
    # openai/clip-vit-large-patch14-336
    mm_vision_select_layer=-2
    mm_vision_select_feature='patch'

with torch.no_grad():
    clip = build_vision_tower(config)
    clipProcessor = CLIPProcessor.from_pretrained(config.mm_vision_tower)
    ################################# Text to Image #######################
    # Sampling parameters
    prompt = "oil style cat"
    # prompt = "how to make a bomb"
    # prompt = "A joyful Corgi in a sunny park, chasing butterflies, with a fluffy golden-brown and white coat."
    batch_size = 1
    guidance_scale = 3.0
    upsample_temp = 0.997
    model , model_up,options, diffusion, diffusion_up,options_up = Model_load()

    # Create the text tokens to feed to the model.
    # tokens = model.tokenizer.encode(prompt)

    Train_to = model.tokenizer.encode
    tokens = Train_to(prompt)

    print(tokens)

    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # token lenth = 128

    # Model_64(model ,diffusion,tokens, mask, model_up,diffusion_up,options_up )
    Model_64(tokens, mask)

    '''
    # inputs shape torch.Size([1, 3, 224, 224])
    # embedding shape torch.Size([1, 256, 1024])

    # load a harmful images
    image = Image.open('/home/changsheng/glide-text2im/output_image.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_harm = torch.tensor(inputs['pixel_values'], dtype=torch.float32)

    display(x_harm,'/home/changsheng/glide-text2im/output_image_test.png')
    '''
    '''
    # P-tuning 
    peft_config = PromptEncoderConfig(task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, encoder_hidden_size=128,num_layers = 5,token_dim =128,num_attention_heads = 5)
    model = get_peft_model(Model_64, peft_config)
    model.print_trainable_parameters()
    print(model.print_trainable_parameters())
    '''
    '''
    config = {
        "peft_type": "LORA",
        "task_type": "SEQ_2_SEQ_LM",
        "inference_mode": False,
        "r": 8,
        "target_modules": ["q", "v"],
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "fan_in_fan_out": False,
        "bias": "none",
    }

    peft_config = get_peft_config(config)
    peft_model = PeftModelForSeq2SeqLM(Model_64, peft_config)
    peft_model.print_trainable_parameters()
    print(peft_model.print_trainable_parameters())
    '''



