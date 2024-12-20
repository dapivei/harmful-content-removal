# util.py

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_anns_on_image(image, anns, alpha=0.7):
    """
    Overlays segmentation masks on the original image with different colors.

    Parameters:
    - image: The original image as a NumPy array.
    - anns: A list of annotations (masks) from SAM.
    - alpha: Transparency factor for the overlay masks (default is 0.7).
    """
    if len(anns) == 0:
        return

    # Sort the annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # Create a copy of the original image to overlay masks
    overlay = image.copy()

    # Create an image for the colored masks
    mask_image = np.zeros_like(image, dtype=np.uint8)

    for ann in sorted_anns:
        segmentation = ann['segmentation']
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        # Apply the color to the mask_image where segmentation is True
        mask_image[segmentation] = color

    # Combine the original image with the mask image using the alpha parameter
    cv2.addWeighted(mask_image, alpha, overlay, 1 - alpha, 0, overlay)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

def batchify(data, batch_size):
    """Split data into batches."""
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

def combine_harmful_masks(image_shape, masks, probs, descriptions, harmful_descriptions):
    """
    Combine masks of harmful segments.

    Args:
        image_shape (tuple): Shape of the original image (height, width).
        masks (list): List of mask dictionaries from SAM.
        probs (list): List of probability arrays from the CLIP model.
        descriptions (list): List of text descriptions used for classification.
        harmful_descriptions (set): Set of descriptions considered harmful.

    Returns:
        np.ndarray: Combined binary mask of harmful regions.
    """
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for mask, prob in zip(masks, probs):
        # Get the index of the highest probability description
        max_index = np.argmax(prob)
        predicted_description = descriptions[max_index]

        # Check if the predicted description is harmful
        if predicted_description in harmful_descriptions:
            # Combine the mask
            combined_mask = np.logical_or(combined_mask, mask['segmentation'])

    combined_mask = combined_mask.astype(np.uint8)
    return combined_mask

def pad_to_square(image):
    h, w, c = image.shape
    size = max(h, w)
    padded_image = np.zeros((size, size, c), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return padded_image

def resize_image(image, size=(224, 224)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

def mask_harmful_content(image, mask, fill_color=(255,255,255), background_color=(0,0,0)):
    """
    Apply a mask to the image, filling the masked area with a specified color.

    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): Binary mask indicating harmful regions.
        fill_color (tuple): RGB color to fill the masked area.
        background_color (tuple): RGB color to fill the background area.

    Returns:
        np.ndarray: The image with harmful content masked.
    """
    masked_image = image.copy()
    masked_image[mask.astype(bool)] = fill_color
    masked_image[~mask.astype(bool)] = background_color
    return masked_image
