# End-to-End Training for Unified Tokenization and Latent Denoising
### Let's UNITE! Single-stage training, Unified model for Tokenization & Generation <br/><br/>[Paper](https://arxiv.org/abs/2603.22283) | [Project Page](https://xingjianbai.com/unite-tokenization-generation/)

This is the code repository of the paper:
> [End-to-End Training for Unified Tokenization and Latent Denoising](https://arxiv.org/abs/2603.22283) (arXiv 2026)  
> [Shivam Duggal*](https://shivamduggal4.github.io/), [Xingjian Bai*](https://xingjianbai.com/), [Zongze Wu](https://betterze.github.io/website/), [Richard Zhang](https://richzhang.github.io/), [Eli Shechtman](https://scholar.google.com/citations?user=B_FTboQAAAAJ&hl=zh-CN), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Phillip Isola](https://web.mit.edu/phillipi/), [William T. Freeman](https://billf.mit.edu/)  
> MIT, Adobe  


## Table of Content
[Abstract](#Abstract)  
[Approach Overview](#Overview)  
[Setup](#Setup)  
[Datasets](#Datasets)  
[Training](#Training)  
[AWM Fine-Tuning](#AWMFineTuning)  
[Pretrained Checkpoints](#PretrainedCheckpoints)  
[Evaluation](#Evaluation)  
[Citation](#Citation)  

<a name="Abstract"></a>
## Abstract

<div style="text-align: justify;">
Latent diffusion models (LDMs) enable high-fidelity synthesis by operating in learned latent spaces. However, training state-of-the-art LDMs requires complex staging: a tokenizer must be trained first, before the diffusion model can be
trained in the frozen latent space. We propose UNITE – an autoencoder architecture for unified tokenization and latent diffusion. UNITE consists of a Generative Encoder that serves as both image tokenizer and latent generator via weight sharing. Our key insight is that tokenization and generation can be viewed as the same latent inference problem under different conditioning regimes: tokenization infers latents from fully observed images, whereas generation infers them from noise together with text or class conditioning. Motivated by this, we introduce a single-stage training procedure that jointly optimizes both tasks via two forward passes through the same Generative Encoder. The shared parameters enable gradients
to jointly shape the latent space, encouraging a “common latent language”. Across image and molecule modalities, UNITE achieves near state of the art performance without adversarial losses or pretrained encoders (e.g., DINO), reaching FID
2.12 and 1.73 for Base and Large models on ImageNet 256 × 256. We further analyze the Generative Encoder through the lenses of representation alignment and compression. These results show that single stage joint training of tokenization & generation from scratch is feasible.
</div>

<a name="Approach Overview"></a>
## Approach Overview
![](./assets/overview.png)

<a name="Setup"></a>
## Setup

```bash
# Clone and install dependencies
mamba create -n unite python=3.10 -y
mamba activate unite
pip install uv

# Install PyTorch 2.10.0 with CUDA 12.8 or higher # or your own cuda version
uv pip install -r requirements.txt
```

Required packages: `torch>=2.1`, `torchvision`, `einops`, `torchdiffeq`, `lpips`, `pyyaml`, `clean-fid`.  
Main requirement is to have a torch version which supports Muon optimizer.

**For FID evaluation on Imagenet 256 x 256:**
```
wget https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth  
wget https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz  
export INCEPTION_WEIGHTS=/path/to/inception_weights  
export IN256_FID_STATS=/path/to/in256_fid_stats  
```

<a name="Datasets"></a>
## Datasets

Training uses the [ImageNet-1K](https://image-net.org/) dataset (ILSVRC2012). Set the `DATA_PATH` environment variable to the training split:

```bash
export DATA_PATH=/path/to/imagenet/train
```

FID evaluation uses precomputed ImageNet statistics from `clean-fid`.  
For our results on the molecules dataset, please refer to the paper.

<a name="Training"></a>
## Training

![](./assets/architecture.png)

Single-node (with 8 GPUs) training procedure on Imagenet 256x256:

```bash
# Using the launch script (auto-detects GPUs and data path)
bash run_scripts/train.sh

# Or directly with torchrun
torchrun --nproc_per_node=8 main_train.py \
    --config configs/imagenet.yaml \
    --data-path $DATA_PATH \
    --experiment-name lets-unite
```

To resume from a checkpoint, additionally provide the path of the checkpoint with argument: `--ckpt`.

Key config fields (see `configs/imagenet.yaml`):

| Field | Default | Description |
|:------|:-------:|:------------|
| `total_batch_size` | 1024 | Effective batch size across all GPUs and accumulation steps |
| `grad_accum_steps` | 2 | Gradient accumulation steps (batch/GPU = total / world_size / accum) |
| `flow_steps_per_recon` | 3 | Flow matching steps per reconstruction forward pass |
| `flow_mini_batch` | 4 | Chunk size for flow steps (controls peak memory) |
| `torch_compile_decoder` | true | torch.compile the decoder for faster training |

**Evaluation during training:**  FID is evaluated automatically at the end of each `fid_epoch` interval (default: every 40 epochs). The evaluation uses adaptive CFG scale search: starting from the current best scale, it tests neighboring scales and updates the best. At `fid_sweep_epoch` intervals (default: 120 epochs), a more comprehensive sweep is performed across multiple CFG intervals and normalization orders.

<a name="AWMFineTuning"></a>
## AWM Fine-Tuning (Experimental)

This repository also includes an Advantage Weighted Matching (AWM) fine-tuning entry point for pretrained ImageNet UNITE checkpoints. The default AWM config rolls out images from class labels, scores them with a frozen reward model, and updates the latent denoiser with an AWM-style flow matching loss while keeping the decoder/tokenizer side frozen as a control experiment. A separate reconstruction variant can also keep the original UNITE reconstruction path active on real ImageNet images.

### Data

AWM-only rollouts do not require an ImageNet dataloader: class labels are sampled uniformly from `0..999`, and multiple images are generated for each label to compute within-class advantages. The reconstruction variant uses real ImageNet images for UNITE-style reconstruction training; provide an extracted `ImageFolder` train directory, `ILSVRC2012_img_train.tar`, or the parent directory containing that tar with `--data-path`, `reconstruction.data_path`, or `DATA_PATH`.

The default config uses:

| Field | Default | Description |
|:------|:-------:|:------------|
| `training.gradient_accumulation_steps` | 4 | Number of denoising/AWM loss microbatches per rollout |
| `training.freeze_decoder` | `true` | Keep the pretrained decoder frozen for the AWM-only control |
| `training.freeze_patch_embed` | `true` | Keep the image patch embedding frozen for the AWM-only control |
| `awm.classes_per_rank` | 8 | Number of ImageNet classes sampled per GPU per rollout |
| `awm.samples_per_class` | 8 | Images sampled per class label; this is the per-rank advantage group size |
| `awm.rollout_micro_batch_size` | 16 | No-grad rollout/reward microbatch size used before the denoising loss accumulation |
| `awm.train_timesteps` | 4 | Number of random flow timesteps trained per generated latent |
| `awm.beta_kl` | 0.05 | Velocity-space KL coefficient to the frozen pretrained UNITE reference |
| `awm.ema_beta` | 1.0 | Velocity-space KL coefficient to the adaptive EMA reference |
| `awm.kl_ema_decay` | 0.3 | Maximum EMA decay used for the adaptive KL reference |
| `sampling.cfg_scale` | 1.0 | Rollout CFG scale; `1.0` keeps rollout and training policy aligned |
| `sampling.grid_interval` | 200 | Save a fixed-class EMA sample grid every N completed optimizer steps |
| `eval.enabled` | `true` | Run distributed 50K-sample FID/IS evaluation when checkpoints are saved |
| `eval.batch_size` | 50 | Per-GPU evaluation generation batch size |
| `reward.model_name` | `dinov2_vitl14_lc` | Frozen DINOv2 ImageNet classifier reward |
| `reward.mode` | `logprob` | Uses `log p_DINOv2(class | image)` as the scalar reward |

For the default 8-GPU launch, each optimizer step samples different labels on each GPU, generates `8 * 8 * 8 = 512` images globally, and computes advantages within each GPU's `8` samples per class. The denoising/AWM loss is then split into `training.gradient_accumulation_steps=4` microbatches, so each backward pass sees `16 * 4 = 64` noised latent/target pairs per GPU while the full optimizer step still uses `512 * 4 = 2048` noised pairs globally.

The reconstruction variant is [configs/imagenet_awm_recon.yaml](configs/imagenet_awm_recon.yaml). It sets `training.freeze_decoder: false`, `training.freeze_patch_embed: false`, and `reconstruction.enabled: true`, so both the encoder/tokenizer path and decoder receive UNITE's L1 + LPIPS reconstruction loss. The reconstruction batch is 64 images per GPU and is split into 16-image backward microbatches after the AWM backward passes, reducing peak activation overlap.

### Checkpoint Preparation

Download a pretrained UNITE checkpoint and set `CKPT_PATH` before launching:

```bash
export CKPT_PATH=/path/to/UNITE-B.pt
```

The default AWM-only config is [configs/imagenet_awm.yaml](configs/imagenet_awm.yaml), which matches the Base encoder + Base decoder checkpoint. The AWM + reconstruction config is [configs/imagenet_awm_recon.yaml](configs/imagenet_awm_recon.yaml). For a Large checkpoint, update the `gen_tok` architecture fields to match the checkpoint before training.

The DINOv2 reward model is loaded through `torch.hub` on the first run. Make sure the training environment can either download the DINOv2 weights or has them already cached. Checkpoint-time FID/IS evaluation uses the same `INCEPTION_WEIGHTS` and `IN256_FID_STATS` environment variables described above; set `eval.enabled: false` to skip it.

### Launch

Single-node 8-GPU AWM-only launch:

```bash
bash run_scripts/train_awm.sh
```

Equivalent direct launch:

```bash
torchrun --nproc_per_node=8 main_train_awm.py \
    --config configs/imagenet_awm.yaml \
    --ckpt "$CKPT_PATH" \
    --results-dir outputs_awm \
    --experiment-name unite-awm-dinov2
```

Single-node 8-GPU AWM + reconstruction launch:

```bash
export DATA_PATH=/path/to/imagenet/train
bash run_scripts/train_awm_recon.sh
```

If ImageNet is only available as the official read-only tar archive, point `DATA_PATH` at either the tar itself or its parent directory. If the archive directory is not writable, set `DATA_INDEX_PATH` to a writable `.npz` path for the tar offset index:

```bash
export DATA_PATH=/path/to/ILSVRC2012_img_train.tar
export DATA_INDEX_PATH=/path/to/writable/imagenet_train_tar_index.npz
```

Checkpoints are written to `outputs_awm/<experiment-name>/checkpoints/` and contain the current policy, checkpoint EMA policy, adaptive KL-EMA policy, optimizer state, and the rollout CFG scale. Fixed-class sample grids are written to `outputs_awm/<experiment-name>/samples/`; `fixed_grid_classes.txt` records the repeated class-id order used in each grid. When `eval.enabled` is true, each checkpoint also triggers distributed EMA sampling for 50K balanced ImageNet labels and logs FID-50K/IS-50K to `outputs_awm/<experiment-name>/eval/metrics.jsonl`. The fine-tuning loss includes two optional stability terms: `awm.beta_kl` keeps the policy close to the frozen pretrained UNITE reference, while `awm.ema_beta` keeps it close to an adaptive EMA reference whose decay follows `awm.kl_ema_decay_type` up to `awm.kl_ema_decay`.

To backfill FID/IS for checkpoints that were saved before checkpoint-time eval was enabled, run:

```bash
torchrun --nproc_per_node=8 eval_awm_checkpoints.py \
    --config configs/imagenet_awm.yaml \
    --checkpoint-dir outputs_awm/unite-awm-dinov2/checkpoints \
    --output-dir outputs_awm/unite-awm-dinov2/eval_offline \
    --state-key ema \
    --skip-existing
```

This loads one checkpoint at a time, evaluates its EMA weights, and appends results to `eval_offline/metrics.jsonl`.

### Reproducing Paper Results
To reproduce the paper results for UNITE-B (with 3 flow mini batches per each reconstruction step)on a single-node on ImageNet-1K 256×256, use the  config
[`configs/imagenet.yaml`](configs/imagenet.yaml).


| | |
|:--|:--|
| **Architecture** | Base encoder (130.6M params) + Base decoder (86.2M params) |
| **Total params** | 217.6M (all trainable) |
| **Hardware** | 1 node × 8 NVIDIA H200 (140 GB each) |
| **Effective batch size** | 1024 (64 per GPU × 2 gradient accumulation steps) |
| **Precision** | BF16 mixed precision |
| **Optimizer** | Muon |
| **Training speed** | ~26 min/epoch |


<br/>

UNITE has an advesarial nature due to joint optimization of reconstruction and denoising losses. See paper Sec. 3.3 for more details. The training curves should be similar to the following graph: 

![](./assets/training_dynamics.png)
<br/>


FID and Inception Score (IS) are computed with adaptive CFG scale search — starting from the current best scale, neighboring scales are evaluated and the best is kept. All evaluations use the EMA model.



![](./assets/fid_curve.png)



<a name="PretrainedCheckpoints"></a>
## Pretrained Checkpoints

| Model | Encoder | Decoder | Total Params | Epochs | FID-50K | Checkpoint |
|:------|:-------:|:-------:|:------------:|:------:|:--:|:----------:|
| United-B | Base | Base | 217.6M | 240 | 2.12 | [Download Link](https://www.dropbox.com/scl/fi/6vbzwqiu9rqnmwm4cj5my/UNITE-B.pt?rlkey=yrqoyiw1zymqe6pn8rf0ss1d6&st=xt8hmz46&dl=0) |
| United-L | Large | Base | 589.0M | 120 | 1.73 | [Download Link](https://www.dropbox.com/scl/fi/mvqbhq3oubog4gtswnthw/UNITE-L.pt?rlkey=ekn3m0b5wb26512dc901rj9w7&st=x91h020h&dl=0) |

Checkpoint includes model weights (encoder + decoder), EMA state, optimizer state, and scheduler state for seamless resumption. Both models were trained with 14 flow iterations per single reconstruction step, so `flow_steps_per_recon = 14`


<a name="Evaluation"></a>
## Evaluation

We will share the inference evaluation for generation and reconstruction FID and IS soon. Till then enjoy the following results from XL model.

![](./assets/generation_visualization.png)
<br/>

<a name="Citation"></a>
## Citation

```bibtex
@article{duggal2026unite,
  title={End-to-End Training for Unified Tokenization and Latent Denoising},
  author={Shivam Duggal and Xingjian Bai and Zongze Wu and Richard Zhang and Eli Shechtman and Antonio Torralba and Phillip Isola and William T. Freeman},
  journal={arXiv preprint arXiv:2603.22283},
  year={2026}
}
```
