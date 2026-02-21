# CraftGraffiti

## Description
Pytorch implementation of the paper [CraftGraffiti: Exploring Human Identity with Custom Graffiti Art via Facial-Preserving Diffusion Models](https://arxiv.org/abs/2508.20640). This model is implemented on top of the [FLUX](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) framework. The proposed model solves the face consistency issue during style transfer in graffiti art in a two-stage process.

<p align="center">
  <img src="https://github.com/ayanban011/CraftGraffiti/blob/main/fig/method.png">
  <be>
<b>CraftGraffiti</b> transforms a source image into a graffiti-style portrait while preserving the subject’s identity and pose. Graffiti style is injected via a pretrained diffusion fine-tuned with LoRA for the dedicated style. Later on, another diffusion model is equipped with face-consistent self-attention and cross-attention modules to preserve key facial features, and a LoRA module enables pose customization without full model retraining via CLIP-based prompt extension. Finally, multi-scale latent feature processing using a VAE ensures that both global structure and fine details are captured across different resolutions in the latent space, yielding a high-quality graffiti-style image.
</p>

## Getting Started

### Step 1: Clone this repository and change the directory to the repository root
```bash
git clone https://github.com/ayanban011/CraftGraffiti.git 
cd CraftGraffiti
```

### Step 2: Setup and activate the conda environment with required dependencies:
```bash
conda env create -f craftgraffiti.yml
```

### Step 3: Download the necessary weights:
- [IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/blob/main/ip-adapter.bin)
- [Graffiti-LoRA](https://civitai.com/models/1058970/graffiti-style-flux)
- [Rendered Face Detailer](https://civitai.com/models/600119/rendered-face-detailer-sdxlflux)

### Step 4: Running
```bash
python main1.py
python main.py
```


## Citation

If you find this useful for your research, please cite it as follows:

```bash
@misc{banerjee2025craftgraffitiexploringhumanidentity,
      title={CraftGraffiti: Exploring Human Identity with Custom Graffiti Art via Facial-Preserving Diffusion Models}, 
      author={Ayan Banerjee and Fernando Vilariño and Josep Lladós},
      year={2025},
      eprint={2508.20640},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.20640}, 
}
```

## Acknowledgement

We have built with the FLUX and IP-Adapter.


## Conclusion
Thank you for your interest in our work, and sorry if there are any bugs.
