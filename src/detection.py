# Import necessary libraries
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from IPython.display import display, HTML
from torchvision import transforms 
import cv2
import pickle
import os
from tqdm import tqdm
# Import utility functions from util.py
from util import (
    show_anns_on_image,
    batchify,
    combine_harmful_masks,
    resize_image,
    mask_harmful_content,
)

def segment_image(image):
    """
    Segments the image using SAM (Segment Anything Model).

    Args:
        image (PIL.Image.Image): Image to segment.

    Returns:
        list: List of segmentation masks.
    """
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,           # Adjust for finer or coarser grid, 64
        min_mask_region_area=50,      # Set minimum area for masks
        box_nms_thresh=0.2,           # Adjust NMS threshold
        stability_score_thresh=0.2,   # Set stability score threshold
    )

    mask_generator.predictor.model.to(device)

    # image_tensor = torch.tensor(image_np).to(device)
    masks = mask_generator.generate(image_np)

    print(f"Generated {len(masks)} masks.")
    return masks

def classify_segments(image, masks, descriptions):
    """
    Classifies each image segment using CLIP.

    Args:
        image (PIL.Image.Image): Original image.
        masks (list): List of segmentation masks.
        descriptions (list): List of descriptions for classification.

    Returns:
        tuple: overall_probs, overall_masks
    """
    # Preprocess segments
    res = []
    for mask in masks:
        segmentation = mask['segmentation']

        # Mask out all other parts
        segment_image = np.array(image).copy()
        mask_bool = segmentation.astype(bool)
        segment_image[~mask_bool] = 255  # Set the background to white

        # Resize the image to 224x224
        segment_resized = resize_image(segment_image)
        res.append(segment_resized)

    # Define batch size
    batch_size = 16

    # Split images into batches
    image_batches = batchify(res, batch_size)

    # Initialize lists to store probabilities and masks
    overall_probs = []
    overall_masks = []

    # Process each batch
    for batch_idx, image_batch in enumerate(image_batches):
        # Process inputs in a batch
        inputs = clip_processor(
            text=descriptions,
            images=image_batch,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Perform inference in a batch
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        # Extract logits and compute probabilities
        logits_per_image = outputs.logits_per_image  # Shape: [batch_size, num_descriptions]
        probs = logits_per_image.softmax(dim=1)      # Shape: [batch_size, num_descriptions]
        
        # Process results for each image in the batch
        for sub_batch_idx, text_probs in enumerate(probs):
            global_image_idx = batch_idx * batch_size + sub_batch_idx  # Absolute image index
            overall_probs.append(text_probs.cpu().numpy())
            overall_masks.append(masks[global_image_idx])
            descs = list(descriptions)  # List of descriptions
            text_probs = text_probs.cpu().numpy() * 100  # Convert to percentages
            
            # Find the index of the maximum probability
            max_index = text_probs.argmax()
            
            # print(f"Segment {global_image_idx + 1}:")
            # for i, (desc, prob) in enumerate(zip(descs, text_probs)):
            #     if i == max_index:
            #         display(HTML(f"<span style='color: red;'>{desc}: {prob:.2f}%</span>"))
            #     else:
            #         print(f"{desc}: {prob:.2f}%")
            # print()
    return overall_probs, overall_masks

def save_masking(original_img, save_path, descriptions, harmful_descriptions):
    original_img = original_img.resize((1024,1024))
    masks = segment_image(original_img)
    
    overall_probs, overall_masks = classify_segments(original_img, masks, descriptions)

    harmful_mask = combine_harmful_masks(
        image_shape=np.array(original_img).shape,
        masks=overall_masks,
        probs=overall_probs,
        descriptions=descriptions,
        harmful_descriptions=harmful_descriptions
    )

    masked_image = mask_harmful_content(np.array(original_img), harmful_mask)
    
    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    # Save the numpy arrays to pickle files
    masked_image_path = os.path.join(save_path, "masked_image.pkl")
    original_image_path = os.path.join(save_path, "original_image.pkl")
    
    # Save the numpy array to a pickle file
    with open(masked_image_path, "wb") as f:
        pickle.dump(masked_image, f)
        
    with open(original_image_path, "wb") as f:
        pickle.dump(original_img, f)

    return original_img, masked_image


# Global variables for models
# Load SAM model
model_type = "vit_h"  # Options: 'vit_h', 'vit_l', 'vit_b'
sam_checkpoint = "../notebooks/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Load CLIP model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


harmful_descriptions = [
    "an image containing nudity",
    "an image containing a woman's breast",
    "an image containing a man's breast",
    "An image that can trigger trypophobia",
    "an image containing a upset face",
    "an image containing a body without any clothes",
    "an image containing a shocking face",
    "an image containing a scary face",
    "an image containing a face with blood",
    "an image containing violence content",
    "an image containing a weird mouth",
    "an image containing a mouth which is large and scary",
    "an image containing blood like material",
    "an image containing a pistol",
    "an image containing a gun",
    "an image containing gore or mutilation",
    "an image containing dismembered body parts",
    "an image depicting animal cruelty",
    "an image showing self-harm",
    "an image containing explicit sexual content",
    "an image promoting hate symbols or offensive gestures",
    "an image containing harmful stereotypes",
    "an image depicting extreme physical injury",
    "an image containing child exploitation content",
    "an image showing graphic medical procedures",
    "an image depicting corpses or dead bodies",
    "an image containing disturbing insect swarms",
    "an image showing natural disasters with devastation",
    "an image with excessive flashing lights (potentially triggering seizures)",
    "an image depicting torture or abuse",
    "an image containing offensive or discriminatory text",
    "an image with disturbing paranormal entities",
    "an image depicting substance abuse or drug paraphernalia",
    "an image showing graphic war scenes or combat",
    "an image containing exaggerated bodily fluids",
    "an image showing facial deformities in a grotesque or harmful manner",
    "an image depicting burning or arson",
    "an image showing sharp objects causing harm",
    "an image of a person in distress or crying excessively",
    "an image of a person being attacked or assaulted",
    "an image of hanging or suicide-related acts",
    "an image showing explicit or obscene gestures",
    "an image containing dangerous animals in an aggressive stance",
    "an image containing hallucination-inducing patterns",
    "an image with distorted or grotesque human features",
    "an image with profane or explicit language",
    "an image showing someone in chains or captivity",
    "an image promoting illegal activities",
    "an image containing depictions of cannibalism",
    "an image showing war crimes or acts of terrorism",
    "an image containing toxic substances or hazardous materials",
    "an image showing diseases or medical conditions in extreme detail",
    "an image of malicious cyber symbols (e.g., hacker threats)",
    "an image showing exaggerated cultural or racial mockery",
    "an image containing disfigured or decomposing animals",
    "an image glorifying extremist propaganda",
    "an image containing unsettling dolls or humanoid figures",
    "an image showing excessive darkness or obscured violence",
    "an image portraying traumatic real-world events",
    "an image with subliminal harmful messaging",
    "an image showing needles or injections in a threatening context",
    "an image showing humans or animals trapped in perilous situations",
    "an image showing full frontal nudity",
    "an image of partial nudity",
    "an image with exposed genitalia",
    "an image containing a naked back",
    "an image showing a naked body partially obscured",
    "an image showing explicit sexual activity",
    "an image of a nude figure in an artistic but explicit context",
    "an image showing bare skin with sexual connotations",
    "an image containing topless individuals",
    "an image of someone wearing see-through clothing",
    "an image with exaggerated focus on nudity-related features",
    "an image depicting underage nudity",
    "an image of someone in a sexually suggestive pose while undressed",
    "an image containing a person wearing minimal clothing inappropriate for the context",
    "an image showing inappropriate body exposure in public settings",
    "an image with implied nudity using shadows or partial coverage",
    "an image of a person covering their body parts in an explicit manner",
    "an image containing pixelated or blurred nudity",
    "an image with lingerie or undergarments depicted in a sexualized manner",
    "an image containing illustrations or animations of nudity",
    "an image with close-ups of body parts typically associated with nudity",
    "an image portraying sexual exploitation or abuse",
    "an image containing nude individuals in a group setting",
    "an image with digitally manipulated nudity",
    "an image showing nudity combined with disturbing elements",
]

safe_descriptions = ["small object that cannot be correctly recognized",
    "an image containing a human being well dressed",
    "an image containing floor",
    "an image containing ground",
    "an image containing playground",
    "an image of a neutral, non-offensive setting, such as a serene natural landscape (a meadow, beach, or clear sky) or a simple indoor space (a clean room with a plain floor and walls). The focus should be calm and peaceful, avoiding any symbols or content that might be seen as harmful or controversial",
]

descriptions = safe_descriptions + harmful_descriptions 



# Ensure these variables are defined before running this code:
# descriptions = ...
# harmful_descriptions = ...

input_dir = '../data/images/baseline'
output_root = '../../save_path_12_19'

# Get the list of image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each file in the baseline directory with a progress bar
for filename in tqdm(image_files, desc="Processing images"):
    # Full path to the input image
    img_path = os.path.join(input_dir, filename)

    # Open the image
    original_img = Image.open(img_path)

    # Extract the image name without extension
    img_name = os.path.splitext(filename)[0]

    # Create a directory named after the image inside the output root
    output_dir = os.path.join(output_root, img_name)
    os.makedirs(output_dir, exist_ok=True)

    # Define the output save path (inside the directory named after the image)
    save_path = output_dir

    # Call the save_masking function with the constructed paths
    org_img, masked_img = save_masking(
        original_img, 
        save_path, 
        descriptions, 
        harmful_descriptions
    )