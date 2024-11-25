from dataset import *
from pipeline import G_StableDiffusionPipeline
from model import GUNet2DConditionModel
from transformers import CLIPTokenizer,CLIPTextModel
from diffusers import DDPMScheduler,AutoencoderKL
from train import train
from sample import sample
import argparse
import random
import numpy as np
import torch
import os
import json
class Main():
    def __init__(self, train_config, env_config, sample_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.sample_config = sample_config
        self.datestr = None
        self.device = env_config['device']
    
        self.gunet = GUNet2DConditionModel(
                    sample_size=train_config['sample_size'],  # the target image resolution
                    in_channels=train_config['in_channel'],  # the number of input channels, 3 for RGB images
                    out_channels=train_config['out_channel'],  # the number of output channels
                    layers_per_block=train_config['layers_per_block'],  # how many ResNet layers to use per UNet block
                    attention_head_dim=train_config['attention_head_dim'],
                    cross_attention_dim=train_config['cross_attention_dim'],
                    mid_block_scale_factor=train_config['mid_block_scale_factor'],
                    freq_shift=train_config['freq_shift'],
                    downsample_padding=train_config['downsample_padding'],
                    block_out_channels=train_config['block_out_channels'],  # the number of output channels for each UNet block
                    down_block_types=(
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                        ),
                    up_block_types=(
                        "UpBlock2D", 
                        "CrossAttnUpBlock2D", 
                        "CrossAttnUpBlock2D", 
                        "CrossAttnUpBlock2D"
                        ),
                    norm_num_groups = train_config['norm_num_groups'],
                    norm_eps=1e-05,
                    flip_sin_to_cos=True,
                    center_input_sample=False,
                    ).to(self.device)
        # self.pretrained = env_config['pretrained_model_name']
        self.pretrained = env_config['pretrained_path']
        self.scheduler = DDPMScheduler.from_pretrained(self.pretrained, subfolder='scheduler')
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained, subfolder='tokenizer')
        self.encoder = CLIPTextModel.from_pretrained(self.pretrained, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(self.pretrained, subfolder='vae')
        
        data_path = self.env_config['data_path']
        self.data_path = data_path
        
        
        with open(self.data_path, 'r') as json_file:
            data = json.load(json_file)
        train_data = CustomDataset(data=data, tokenizer=self.tokenizer)
        loader = torch.utils.data.DataLoader(train_data,
                                        shuffle=True,
                                        collate_fn=collate_fn,
                                        batch_size=train_config['batch_size'])
        self.loader = loader
    def run(self):
        if self.env_config['mode'] == 'train':
            
            self.train_log = train(self, gunet=self.gunet, 
                                   scheduler=self.scheduler, 
                                   tokenizer=self.tokenizer, 
                                   encoder=self.encoder, 
                                   vae=self.vae, 
                                   train_config = self.train_config,
                                   env_config = self.env_config,
                                   dataloader = self.loader
                                )
        elif self.env_config['mode'] == 'sample':
            epoch = env_config['sample_epoch']
            device = env_config['device']
            save_path = os.path.join(self.env_config['save_path'],f'save_{epoch}')
            model_name = env_config['trained_model_name']
            # model = G_StableDiffusionPipeline.from_pretrained(
            #     save_path,
            #     safety_checker=None).to(device)
            model = G_StableDiffusionPipeline.from_pretrained(
                model_name,
                safety_checker=None).to(device)
            prompt = self.sample_config['prompt']
            negative_prompt = self.sample_config['negative_prompt']
            height = self.sample_config['height']
            width = self.sample_config['width']
            num_inference_steps = self.sample_config['num_inference_steps']
            num_images_per_prompt = self.sample_config['num_images_per_prompt']
            sample(self, model=model,
                 prompt=prompt,
                 negative_prompt=negative_prompt,
                 height=height,
                 width=width,
                 num_inference_steps=num_inference_steps,
                 num_images_per_prompt=num_images_per_prompt,
                 sample_config = self.sample_config,
                 env_config = self.env_config,
                 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', help='the target image resolution', type=int, default=64)
    parser.add_argument('--in_channel', help='the number of input channels, 3 for RGB images, 17 for heatmaps', type=int, default=17)
    parser.add_argument('--out_channel', help='the number of output channels', type=int, default=17)   
    parser.add_argument('--layers_per_block', help='how many ResNet layers to use per UNet block', type=int, default=2) 
    parser.add_argument('--attention_head_dim', help='attention head dim', type=int, default=8)
    parser.add_argument('--cross_attention_dim', help='cross attention dim', type=int, default=768)
    parser.add_argument('--mid_block_scale_factor', help='midblock scale factor', type=int, default=1)
    parser.add_argument('--freq_shift', help='', type=int, default=0)
    parser.add_argument('--downsample_padding', help='', type=int, default=1)
    parser.add_argument('--block_out_channels', help='the number of output channels for each UNet block', type=int, nargs='+', default=[320, 640, 1280, 1280])
    parser.add_argument('--norm_num_groups', help='norm num groups', type=int, default=32)
    parser.add_argument('--epoch', help='train epoch', type=int, default=1000)
    
    parser.add_argument('--random_seed', help='random seed', type = int, default=0)
    
    parser.add_argument('--batch_size', help='batch size', type=int, default=17)
    parser.add_argument('--sample_epoch', help='the checkpoin for sample', type=int, default=1000)
    parser.add_argument('--data_path', help='data path', type=str, default='/data/train.json')
    parser.add_argument('--save_path', help='checkpoin save path', type=str, default='/trained')
    parser.add_argument('--pretrained_path', help='pretrained model path', type=str, default='stable-diffusion-v1-5')
    parser.add_argument('--pretrained_model_name', help='', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    # parser.add_argument('--trained_model_name', help='', type=str, default='sw1125/gunet')
    parser.add_argument('--trained_model_name', help='', type=str, default='/trained/save_1000')
    
    parser.add_argument('--device', help='cuda / cpu', type=str, default='cuda:5')
    parser.add_argument('--mode', help='train / sample', type=str, default='train')
    
    
    parser.add_argument('--sample_prompt', help='prompt for sample', type=str, default='A girl is eating an apple.')
    parser.add_argument('--negative_prompt', help='negative prompt for sample', type=str, default='')
    parser.add_argument('--height', help='height for sample', type=int, default='256')
    parser.add_argument('--width', help='width for sample', type=int, default='256')
    parser.add_argument('--num_images_per_prompt', help='', type=int, default='17')
    parser.add_argument('--num_inference_steps', help='', type=int, default='27')
    parser.add_argument('--sample_image_save_path', help='sample_image_save_path', type=str, default='/samples')
    
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    
    train_config = {
        'sample_size': args.sample_size,
        'in_channel': args.in_channel,
        'out_channel': args.out_channel,
        'layers_per_block': args.layers_per_block,
        'attention_head_dim': args.attention_head_dim,
        'cross_attention_dim': args.cross_attention_dim,
        'mid_block_scale_factor': args.mid_block_scale_factor,
        'freq_shift': args.freq_shift,
        'downsample_padding': args.downsample_padding,
        'block_out_channels': tuple(args.block_out_channels),
        'norm_num_groups': args.norm_num_groups,
        'pretrained_model_name': args.pretrained_model_name,
        'pretrained_path': args.pretrained_path,
        'epoch': args.epoch,
        'batch_size': args.batch_size,
    }
    
    env_config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'pretrained_path': args.pretrained_path,
        'device': args.device,
        'trained_model_name': args.trained_model_name,
        'mode': args.mode,
        'random_seed': args.random_seed,
        'sample_epoch': args.sample_epoch,
        
    }
    
    sample_config = {
        'prompt': args.sample_prompt,
        'negative_prompt': args.negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_images_per_prompt': args.num_images_per_prompt,
        'num_inference_steps': args.num_inference_steps,
        'sample_image_save_path': args.sample_image_save_path,
    }
    
    main = Main(train_config, env_config, sample_config, debug=False)
    main.run()