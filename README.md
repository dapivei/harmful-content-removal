# Zero-Shot Harmful Content Removal in Text-to-Image Generation

**Course Project**: CSCI-GA.2271 (Computer Vision)  
**Institution**: New York University

**Authors**:  
- Yuxuan Chen ([yc7087@nyu.edu](mailto:yc7087@nyu.edu))  
- Daniela Pinto Veizaga ([dp3766@nyu.edu](mailto:dp3766@nyu.edu))  
- Amelia Dai ([hd2584@nyu.edu](mailto:hd2584@nyu.edu))  
- De Ren ([dr3702@nyu.edu](mailto:dr3702@nyu.edu))

---

## Overview

This project explores zero-shot harmful content removal in text-to-image generation. Our aim is to detect and remove harmful or sensitive elements from generated images, ensuring safe and controlled outputs without manual curation of harmful prompts.

---

## Folder Structure

**data/**
- **classifier/**: Contains prediction results obtained from Q16 and NudeNet classifiers.
- **images/**: Contains generated images from baseline models, controlled methods, and our proposed approach. Folder 'images/inpaint_new_mask_1219_revised_prompt' contains the final results that we report.
- **prompts/**: Includes the prompt texts used to generate potentially harmful images.

**notebooks/**
- **classifiers.ipynb**: Notebook to run and collect classification results using Q16 and NudeNet.
- **detection.ipynb**: Example notebook for segmenting images and detecting harmful segments.
- **images.ipynb**: Notebook for generating harmful images.
- **prompts.ipynb**: Notebook for collecting and managing harmful prompts.
- **summary.ipynb**: Contains a summary and presentation of experimental results.
- **inpainting.ipynb**: Notebook for inpainting and visualizing the inpainted results.
- **revise prompt.ipynb**: Notebook for getting safe revised prompt using LLM in the inpainting step.
- **ssim.ipynb**: Calculating the semantic similarity.

**src/**  
- Contains executable scripts for harmful segment detection and inpainting.

---

## Getting Started

1. **Installation**:  
   - Ensure you have a suitable Python environment set up.  
   - Install dependencies listed in `requirements.txt`.

2. **Harmful Content Detection & Removal**:  
   - Use scripts in `src/detection.py` to detect harmful segments in generated images.  
   - Employ the inpainting methods in `src/inpainting.py` to remove these segments seamlessly.

---

## Contact

For questions or further information, please reach out to the authors via the provided email addresses.

---  
