from pipeline import G_StableDiffusionPipeline
from utils.draw_pose import *

def sample(self,model=None,
            prompt=None,
            negative_prompt=None,
            height=256,
            width=256,
            num_inference_steps=27,
            num_images_per_prompt=1,
            sample_config={},
            env_config={}):
    self.sample_config = sample_config
    device = env_config['device']
    output = model(prompt=prompt,
                 negative_prompt=negative_prompt,
                 height=height,
                 width=width,
                 num_inference_steps=num_inference_steps,
                 num_images_per_prompt=num_images_per_prompt)
    image_save_path = self.sample_config['sample_image_save_path']
    for im in range(num_images_per_prompt):
        image = output[0][im]
        keypoint_coor = get_coor(image)
        subset = get_subs(channel_num=17, coordinates=keypoint_coor)
    
        canvas = draw_bodypose(np.array(keypoint_coor), np.array([subset]))
        canvas = np.array((canvas - np.min(canvas)) / (np.max(canvas) - np.min(canvas)))
        plt.imsave(os.path.join(image_save_path,f"{im}.png"), canvas[:, :, [2, 1, 0]])
    
    
    