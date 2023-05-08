# LRA

## Luminous pin of [https://github.com/OpenNLPLab/lra](lra)

The codebase for LRA experiments, which is based on [S4](https://github.com/HazyResearch/state-spaces). 

### How to add a new model and experiment:

 - In `src/models/sequence/tnn_draft/`: add `<model_name>.py` (e.g. `skitno.py`)
 - Edit `src/utils/registry.py`: add line to `layers` dict
 - In `configs/model/`: add `<model_name>.yaml` (e.g. `skitno.yaml`)
 - In `configs/experiment/`: add `<model_name>-lra-<dataset_name>.yaml` (e.g. `skitno2d-lra-cifar.yaml`)
 - At `src/models/sequence/model.py:L192`: add `<model_name>` to the `list`.

Notes:
 - For now (2023-05-01), need to run `pip install -e .` from the `fairseq-tnn/` dir while in the `lra` Conda environment to properly load our modules. We can refactor to remove this dependency (probably `laxtnn/modules/__init__.py`). Afterwards, need to reinstall `hydra-core==1.2.0` as well