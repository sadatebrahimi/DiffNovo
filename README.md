
# DiffNovo
## A Transformer-Diffusion Model for De Novo Peptide Sequencing.
DiffNovo is an innovative tool for de novo peptide sequencing using advanced machine learning techniques. This guide will help you get started with installation, dataset preparation, and running key functionalities like model training, evaluation, and prediction.

Ebrahimi, Shiva, et al.'DiffNovo: A Transformer-Diffusion Model for De Novo Peptide Sequencing'

---

## Installation

To manage dependencies efficiently, we recommend using [conda](https://docs.conda.io/en/latest/). Start by creating a dedicated conda environment:

```sh
conda create --name diffnovo_env python=3.10
```

Activate the environment:

```sh
conda activate diffnovo_env
```

Install DiffNovo and its dependencies via pip:

```sh
pip install diffnovo
```

To verify a successful installation, check the command-line interface:

```sh
diffnovo --help
```

---

## Dataset Preparation

### Download DIA Datasets

Annotated DIA datasets can be downloaded from the [datasets page](https://github.com/Biocomputing-Research-Group/DiffNovo/diffnovo-main/datasets).

---

### Download Pretrained Model Weights

DiffNovo requires pretrained model weights for predictions in `denovo` or `eval` modes. Compatible weights (in `.ckpt` format) can be found on the [pretrained models page](https://zenodo.org/records/14611534?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk1YzRhNWEzLTJmYzgtNDA1YS05OWM3LTgyMjhlMzQ1YWRlOSIsImRhdGEiOnt9LCJyYW5kb20iOiI0N2NjZWYwNDdkZjU4NWFhYjJjZTg3NGIzMTY0MGU5NCJ9.PrI1Q_CLJ-ps3xx95PrBe3u5npVI0GAgdbLGhb7dvyp4sIktxK5g3_GLFNLB-gOoQDmJJjQwvlNM72FWmHKVeQ).

Specify the model file during execution using the `--model` parameter.


---

## Usage

### Predict Peptide Sequences

DiffNovo predicts peptide sequences from MS/MS spectra stored in MGF files. Predictions are saved as a CSV file:

```sh
diffnovo --mode=denovo --model=pretrained_checkpoint.ckpt --peak_path=path/to/spectra.mgf
```

---

### Evaluate *de novo* Sequencing Performance

To assess the performance of *de novo* sequencing against known annotations:

```sh
diffnovo --mode=eval --model=pretrained_checkpoint.ckpt --peak_path=path/to/test/annotated_spectra.mgf
```

Annotations in the MGF file must include peptide sequences in the `SEQ` field.

---

### Train a New Model

To train a new DiffNovo model from scratch, provide labeled training and validation datasets in MGF format:

```sh
diffnovo --mode=train --peak_path=path/to/train/annotated_spectra.mgf --peak_path_val=path/to/validation/annotated_spectra.mgf
```

MGF files must include peptide sequences in the `SEQ` field.

---

### Fine-Tune an Existing Model

To fine-tune a pretrained DiffNovo model, set the `--train_from_scratch` parameter to `false`:

```sh
diffnovo --mode=train --model=pretrained_checkpoint.ckpt \
 --peak_path=path/to/train/annotated_spectra.mgf \
 --peak_path_val=path/to/validation/annotated_spectra.mgf
```

---

For further details, refer to our documentation or raise an issue on our [GitHub repository](https://github.com/Biocomputing-Research-Group/DiffNovo/issues). 


# DiffNovo
