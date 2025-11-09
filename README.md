# Set Consistency Energy Networks

**Official code for the ACL 2025 Oral paper:**  
_[Introducing Verification Task of Set Consistency with Set-Consistency Energy Networks]_  
**ACL Anthology:** https://aclanthology.org/2025.acl-long.1599/ · **arXiv:** https://arxiv.org/abs/2503.10695


## Environment

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate set_consistency
```

If PyTorch installation fails, install the GPU-enabled build following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

## Datasets

You can download the prepared dataset from Google Drive:  
👉 [Download dataset (Google Drive)](https://drive.google.com/file/d/1vF19VRmwjQd5BqdjrzE5sP05h6hHzp4Z/view?usp=sharing)

## Model Weights

You can download the trained model from Google Drive:  
👉 **[Download model (Google Drive)](https://drive.google.com/file/d/1w81S9Ut6Fg3EtYz4F_f99bJN0uhcXbb1/view?usp=sharing)**

## Training

Example command for training an energy model:

```bash
  python train_wdb_Set_Contrastive.py \
    --task vqa \
    --dataset lconvqa \
    --loss_type margin \
    --decomposition no \
    --repre_model roberta
```

The resulting model is stored in `results/task/dataset/job_id/`.

## Evaluation

To evaluate a trained energy model:

```bash
  python evaluate.py --task vqa --dataset lconvqa --loss_type margin --decomposition no --repre_model roberta
```

Baseline models can be evaluated with:

```bash
  python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-4o-mini --shot_num 5
```

## Probing Classifiers

The `probing` module adds hidden-state extraction and multiple probe training routines:

1. **Hidden-state cache**  
   Run the local LLM once per example and cache the final-token activations (all layers, both `consistent` and `inconsistent` completions):  
   ```bash
   python probing/extract_hidden_states.py --task vqa --dataset lconvqa --config config.yaml --probe_split train
   ```
   Cached tensors are stored under `results/probing_cache/<task>/<dataset>/<model>/<split>/`.

2. **Supervised probes (linear & MLP)**  
   After caching, launch hyperparameter sweeps with optional parallel workers:  
   ```bash
   python probing/train_supervised.py --task vqa --dataset lconvqa \
       --probe_train_split train --probe_eval_split test --probe_types linear,mlp
   ```
   Results for every sweep configuration are saved as JSON in `results/probing_cache/.../probe_runs/`.

3. **Unsupervised CCS probe**  
   Implemented following the Contrast-Consistent Search objective (Burns et al., 2024).  
   ```bash
   python probing/train_unsupervised_ccs.py --task vqa --dataset lconvqa \
       --probe_train_split train --probe_eval_split test
   ```
   The CCS sweeps share the same cache and prompt configuration as the supervised runs, so no extra LM forward passes are necessary.

All probing scripts reuse the prompts from baseline decoding and read their sweeps/defaults from the `probing` section of `config.yaml`.

## Locate

To run the locate procedure on a trained energy model:

```bash
  python locate.py --task vqa --dataset lconvqa --loss_type margin --decomposition no
```

## SLURM Example

Example `sbatch` script for a single GPU on Linux:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

module load anaconda
conda activate set_consistency
  python train_wdb_Set_Contrastive.py --task vqa --dataset lconvqa --loss_type margin --decomposition no --repre_model roberta
```

## License

This project is released under the MIT License.
