# Spatial Reasoning with Denoising Models
<a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Bart.html">Bartlomiej Pogodzinski</a>, <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele">Bernt Schiele</a>, <a href="http://geometric-rl.mpi-inf.mpg.de/people/Lenssen.html">Jan Eric Lenssen</a>

Max Planck Institute for Informatics, Saarland Informatics Campus

[![arXiv](https://img.shields.io/badge/arXiv-2403.16292-b31b1b.svg)]()
[![Project Website](https://img.shields.io/badge/Website-Visit%20Here-006c66)](https://geometric-rl.mpi-inf.mpg.de/srm/)

## Contents
- [🌐 Project Website](https://geometric-rl.mpi-inf.mpg.de/srm/)
- [📓 Abstract](#-abstract)
- [🛠️ Installation](#️-installation)
- [💾 Datasets & Checkpoints](#-datasets--checkpoints)
- [📣 Usage](#-usage)
- [📘 Citation](#-citation)

## 📓 Abstract
We introduce Spatial Reasoning Models (SRMs), a framework to perform  reasoning over sets of continuous variables via denoising generative models. SRMs infer continuous representations on a set of unobserved variables, given observations on observed variables. Current generative models on spatial domains, such as diffusion and flow matching models, often collapse to hallucination in case of complex distributions. To measure this, we introduce a set of benchmark tasks that test the quality of complex reasoning in generative models and can quantify hallucination. The SRM framework allows to report key findings about importance of sequentialization in generation, the associated order, as well as the sampling strategies during training. It demonstrates, for the first time, that order of generation can successfully be predicted by the denoising network itself. Using these findings, we can increase the accuracy of specific reasoning tasks from <1% to >50%.

## 🛠️ Installation

To get started, create a virtual environment using Python 3.12+:

```bash
python3.12 -m venv srm
source srm/bin/activate
pip install -r requirements.txt
```

## 💾 Datasets & Checkpoints

### Datasets
We provide the relevant files for the datasets as part of our releases [here](https://github.com/Chrixtar/latentsplat/releases).
Please extract the `datasets.zip` in the project root directory or modify the root path of the dataset config files in `config/dataset`.
For counting polygons on FFHQ background, please download [FFHQ](https://github.com/NVlabs/ffhq-dataset) first and provide the path in `config/dataset/counting_polygons_ffhq.yaml`.

### Checkpoints
We provide checkpoints of all trained models in our releases [here](https://github.com/Chrixtar/latentsplat/releases).
Simply download all and extract them in the project root directory.

## 📣 Usage

We have two different settings for debugging (running offline and including typechecking at runtime) and fast training (including torch.compile and wandb logging) and sampling (deactivated typechecking). Use `[debug_](train | test).sh` for training/testing with/without debugging mode.

### Training

Start training via `train.sh` like:

```bash
bash train.sh [experiment config name] [optional experiment id] [optional hydra overrides]
```
, where 
* experiment config name is the file name of the experiment config in `config/experiment` without extension, 
* experiment id (datetime as default) is the optional id of a previous training run to resume (given in `outputs/[experiment config name]/[experiment id]`), and
* hydra overrides for individual hyperparameters can be specified as described [here](https://hydra.cc/docs/advanced/override_grammar/basic/).


The training code will automatically run in distributed mode on all available GPUs, if there are multiple.

### Evaluation

To run evaluation, use `test.sh` like:

```bash
bash test.sh [experiment config name] [experiment id] [test config name] [optional hydra overrides]
```
, where all arguments are the same as for training except for test config name being the file name of the test config in `config/test` without extension. Note that the test script loads the checkpoints from `outputs/[experiment config name]/[experiment id]/checkpoints/last.ckpt`. Evaluation outputs are stored in `outputs/[experiment config name]/[experiment id]/test`.

For example, after downloading our datasets and checkpoints, run the following command for our best setup on the hard difficulty of the MNIST Sudoku dataset:
```bash
bash test.sh ms1000_28 paper ms_hard_seq_adaptive000
```

## 📘 Citation
When using this code in your project, consider citing our work as follows:
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{wewer25srm,
    title     = {Spatial Reasoning with Denoising Models},
    author    = {Wewer, Christopher and Pogodzinski, Bartlomiej and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {arXiv},
    year      = {2025},
}</code></pre>
  </div>
</section>

## Acknowledgements

This project was partially funded by the Saarland/Intel Joint Program on the Future of Graphics and Media. We thank Thomas Wimmer for proofreading and helpful discussions. 
