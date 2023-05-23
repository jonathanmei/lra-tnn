# LRA experiments for TNN (from "SKI to go Faster")

A submodule to run Long-Range Arena experiments for TNN, SKI-TNN, and FD-TNN, implementing ["SKI to go Faster: Accelerating Toeplitz Neural Networks via Asymmetric Kernels."](https://arxiv.org/abs/2305.09028)

The `main` branch is the minimal code that reproduces the paper. The `dev` branch is derived from a custom pin of [lra](https://github.com/OpenNLPLab/lra), which is itself based on [S4](https://github.com/HazyResearch/state-spaces). The `dev` branch allows running experiments on a larger set of architectures.

## Example of how to add a new TNN variant and experiment:

 - In `src/models/sequence/tnn_draft/`: add `<model_name>.py` (e.g. `skitno.py`)
 - Edit `src/utils/registry.py`: add line to `layers` dict
 - In `configs/model/`: add `<model_name>.yaml` (e.g. `skitno.yaml`)
 - In `configs/experiment/`: add `<model_name>-lra-<dataset_name>.yaml` (e.g. `skitno2d-lra-cifar.yaml`)
 - At `src/models/sequence/model.py:L192`: add `<model_name>` to the `list`.

Notes:
 - For now (2023-05-01), need to run `pip install fairseq` (can be official version or [custom pin](https://github.com/jonathanmei/fairseq-tnn)) while in the `lra` Conda environment to properly load our modules. We can refactor to remove this dependency as future work. Afterwards, need to reinstall `hydra-core==1.2.0` as well