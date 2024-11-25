from model import GUNet2DConditionModel
from pipeline import G_StableDiffusionPipeline
from utils.env import *
from tqdm import tqdm


def get_loss(data, encoder, scheduler, gunet):
    device = data['input_ids'].device
    
    out_encoder = encoder(data['input_ids'])[0]
    out_vae = data['pixel_values'] 

    noise = torch.randn_like(out_vae)
    
    noise_step = torch.randint(0, 1000, (1, )).long()
    noise_step = noise_step.to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    out_unet = gunet(out_vae_noise, noise_step, out_encoder).sample
    
    mse_loss = torch.nn.MSELoss()
    
    return mse_loss(out_unet, noise)


def train(self, gunet=None,scheduler=None,tokenizer=None,encoder=None,vae=None,
          train_config={},env_config={},dataloader=None):
    
    self.scheduler = scheduler
    self.tokenizer = tokenizer
    self.vae = vae
    self.encoder = encoder
    self.gunet = gunet
    
    self.encoder.requires_grad_(True) 
    
    self.gunet.train()
    
    seed = env_config['random_seed']
    optimizer = torch.optim.AdamW(gunet.parameters(),
                                lr=1e-5,
                                betas=(0.9, 0.999),
                                weight_decay=0.01,
                                eps=1e-8)
    loader = dataloader
    device = env_config['device']
    epochs = train_config['epoch']
    batch_size = train_config['batch_size']
    pretrained_model_name = train_config['pretrained_model_name']
    save_path  = env_config['save_path']
    loss_sum = 0
    self.encoder.to(device)
    self.gunet.to(device)
    self.vae.to(device)
    for epoch in range(epochs):
        with tqdm(total=len(loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=False) as pbar:
            for i, data in enumerate(loader):
                for k in data.keys():
                    data[k] = data[k].to(device)
                    
                loss = get_loss(data, self.encoder, self.scheduler, self.gunet) 
                loss.backward()
                loss_sum += loss.item()

                if (epoch * len(loader) + i) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(gunet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.update(1)  
                
        if epoch % 1 == 0:
            print(epoch, loss_sum / batch_size)
            loss_sum = 0
            
        if epoch % 100 == 0:
            G_StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name, text_encoder=encoder, vae=vae,
                    unet=gunet).save_pretrained(os.path.join(save_path, f"{epoch}"))
            
            
