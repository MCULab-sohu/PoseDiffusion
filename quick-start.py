from pipeline import G_StableDiffusionPipeline
from utils.draw_pose import *

from diffusers import UNet2DConditionModel
from model import GUNet2DConditionModel

device = "cuda:5"
model = G_StableDiffusionPipeline.from_pretrained('/trained/save_1000',safety_checker=None).to(device)
gunet = GUNet2DConditionModel.from_pretrained("/trained/save_1000/unet").to(device)
model.unet = gunet

# model = G_StableDiffusionPipeline.from_pretrained('sw1125/gunet',safety_checker=None).to(device)
# gunet = GUNet2DConditionModel.from_pretrained("sw1125/gunet_unet").to(device)
# model.unet = gunet

# text = 'A man riding a surfboard on top of a wave in the ocean.'
text = 'Girl drinking coffee'
num_images = 1
output = model(prompt=text,negative_prompt=None,height=256,width=256,num_inference_steps=27,num_images_per_prompt=num_images)
for im in range(num_images):
    image = output[0][im]
    keypoint_coor = get_coor(image)
    subset = get_subs(channel_num=17, coordinates=keypoint_coor)
    canvas = draw_bodypose(np.array(keypoint_coor), np.array([subset]))
    canvas = np.array((canvas - np.min(canvas)) / (np.max(canvas) - np.min(canvas)))
    plt.imsave((f"/samples/{im}.png"), canvas[:, :, [2, 1, 0]])
