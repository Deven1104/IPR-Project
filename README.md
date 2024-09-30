# IPR-Project


# Multimodal Garment Designer: Human-Centric Latent Diffusion Models for Fashion Image Editing

This repository contains the code and datasets for the paper "Multimodal Garment Designer: Human-Centric Latent Diffusion Models for Fashion Image Editing" by Baldrati et al. This work introduces a novel approach to human-centric fashion image generation conditioned on multimodal inputs, such as text descriptions, human body poses, and garment sketches, utilizing a latent diffusion model.

## Introduction

Fashion image editing traditionally involves text-based generation or virtual try-ons. This paper addresses the limitations of previous methods by proposing the **Multimodal Garment Designer (MGD)**, a latent diffusion model capable of generating high-quality, human-centric fashion images. The model is conditioned on multiple input modalities (text, pose, sketch), which makes it more flexible and effective in generating realistic fashion designs while maintaining the original identity and body shape of the subject. This approach represents a leap from prior GAN-based approaches, focusing instead on latent diffusion models (LDMs), known for their superior generative capabilities and efficiency in latent space. 

Additionally, the authors extend two fashion datasets, **Dress Code** and **VITON-HD**, with multimodal annotations, which include textual descriptions and garment sketches, to support this task.

## Key Methodologies

### 1. **Multimodal-Conditioned Latent Diffusion Model**
The core model is based on **Stable Diffusion**, which is modified to take into account multiple conditions:
   - **Textual descriptions (Y)** using a CLIP text encoder.
   - **Pose map (P)**, representing the human keypoints to retain the body pose.
   - **Garment sketch (S)**, representing spatial features of the clothing item.

The model is trained to denoise the latent variables conditioned on these inputs using a modified U-Net architecture. The optimization objective is defined as:

$$
L = E_{E(I),Y,\epsilon \sim N(0,1),t} \left[ \lVert \epsilon - \epsilon_\theta(\gamma, \psi) \rVert_2^2 \right]
$$

where:
- $\gamma$ represents the spatial input (latent variable).
- $\psi$ represents the conditioning input (text, pose, sketch).
- $\epsilon$ is the added Gaussian noise.
  
### 2. **Inpainting Strategy**
A human-centric inpainting process is employed, where the model focuses on replacing the target garment while preserving the subject's identity and body information. The inpainting process concatenates the masked image, the binary inpainting mask, and the encoded latent features.

The denoising process is described by the following objective function:

$$
L = E_{E(I),Y,\epsilon \sim N(0,1),t,E(I_M),m,p,s} \left[ \lVert \epsilon - \epsilon_\theta(\gamma, \psi) \rVert_2^2 \right]
$$

where $I_M$ is the masked input image, and \(p\) and \(s\) represent pose and sketch conditioning, respectively.

### 3. **Classifier-Free Guidance**
To improve the generation quality, the model utilizes classifier-free guidance. The combined noise prediction is given by:

$$
\hat{\epsilon}_\theta(z_t|c) = \epsilon_\theta(z_t|\emptyset) + \alpha (\epsilon_\theta(z_t|c) - \epsilon_\theta(z_t|\emptyset))
$$

where:
- $\epsilon_\theta(z_t|\emptyset)$ is the unconditioned prediction.
- $\epsilon_\theta(z_t|c)$ is the conditioned prediction.
- $\alpha$ controls the guidance scale.

### 4. **Data and Training**
The model is trained on **Dress Code** and **VITON-HD** datasets, extended with textual and sketch annotations. The training uses a combination of pose, text, and sketch inputs to generate fashion images, with additional masking techniques to preserve the model's identity.

## Conclusion

The **Multimodal Garment Designer (MGD)** demonstrates a novel and effective method for human-centric fashion image generation by leveraging latent diffusion models conditioned on multimodal inputs. It outperforms existing methods both quantitatively and qualitatively, as shown through extensive experiments and human evaluations. This work opens up new possibilities for integrating AI into the fashion design process, making it a valuable tool for designers and creative industries.

---

For more details, please refer to the full paper [here](https://github.com/aimagelab/multimodal-garment-designer).

