import torch
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from torchvision import transforms 
import cv2
import os
import pickle
from tqdm import tqdm
from diffusers import StableDiffusion3InpaintPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusion3InpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.to("cuda")


hate_conce_dict = {
    "hate": ["hate"],
    "harassment": ["mocking", "bullying", "threatening", "harassment"],
    "violence": ["humiliation", "violence acts", "suffering", "violence"],
    "self-harm": ["suicide", "eating disorder", "self-harm", "attempts at harming oneself"],
    "sexual": ["sexual acts", "content meant to arouse sexual excitement", "sexual", "nudity"],
    "shocking": ["profane subjects that may shock or disgust", "bodily fluids", "shocking", "obscene gestures"],
    "illegal activity": ["vandalism", "illegal activity", "drug use", "theft"]
}
combined_list = [item for sublist in hate_conce_dict.values() for item in sublist]
NEGATIVE_PROMPT = "" + ', '.join(combined_list) + ""


def inpaint(image, mask, prompt):
    negative_prompt = NEGATIVE_PROMPT #"nude, gun, distorted, grotesque, offensive, violent, unsafe, unethical, horror, creepy"
    image = pipe(
                prompt=prompt, 
                image=image, 
                mask_image=mask,
                strength=0.9, # controls how much of the masked area is regenerated, 0-1, 1 is the most deviated from the origin
                # guidance_scale=10.0  # controls how closely the model follows the prompt, default = 7
                negative_prompt = negative_prompt
            ).images[0]
    return image

def get_bounding_box_masks(mask):
    mask = mask.convert("L")
    mask_np = np.array(mask)

    # Ensure binary mask (thresholding if needed)
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

    # Find contours (connected components)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    bounding_box_mask = Image.new("L", mask.size, 0)  # Black background

    # Draw all bounding boxes on the blank mask
    draw = ImageDraw.Draw(bounding_box_mask)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        draw.rectangle([x, y, x + w, y + h], fill=255)  # Fill the bounding box with white

    return bounding_box_mask

def get_dilated_mask(mask):
    mask = mask.convert("L")
    mask_np = np.array(mask)
    
    kernel_size = 10  # Adjust the kernel size to control enlargement
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=3)  # More iterations enlarge further
    dilated_mask = Image.fromarray(dilated_mask_np)
    return dilated_mask


if __name__ == '__main__':
    # get image and mask
    path = '/scratch/yc7087/saved_image_masks'
    img_id_ls = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            img_id_ls.append(dir_name) #os.path.join(root, dir_name)
    
    # get prompt
    prompts_df = pd.read_csv("../data/prompts/lexica_hate_prompts.csv")

    for img_id in tqdm(img_id_ls):
        mask_path = os.path.join(path, img_id, 'masked_image.pkl')
        with open(mask_path, "rb") as f:
            mask = pickle.load(f)
            mask = Image.fromarray(mask)

        image_path = os.path.join(path, img_id, 'original_image.pkl')
        with open(image_path, "rb") as f:
            image = pickle.load(f)

        original_prompt = prompts_df.loc[prompts_df['unique_id'] == img_id, 'prompt'].iloc[0]
        # prompt = f"Fill the masked area with UNHARMFUL content, but still try to maintain the requirement in '{original_prompt}'"
        prompt = f"Fill the masked area with SAFE and UNHARMFUL content, aligning with the context and artistic intent described in '{original_prompt}', while avoiding any explicit, harmful, or inappropriate elements."

        bounding_box_mask = get_bounding_box_masks(mask)
        dilated_bounding_box_mask = get_dilated_mask(bounding_box_mask)
        inpaint_image = inpaint(image, dilated_bounding_box_mask, prompt)

        inpaint_image.save(f"../data/images/inpaint/{img_id}.png")


