# MEMO-H504-2425 : 

This project explores the use of NeRF (Neural Radiance Fields) based on data captured from a plenoptic camera. Its goal is to synthesize realistic 3D views by exploiting the angular and spatial richness of plenoptic images.

It is built upon [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), a PyTorch implementation of the original NeRF model.

## Installation

```
git clone https://github.com/ijellal/MEMO-H504-2425.git
cd MEMO-H504-2425
pip install -r requirements.txt
```
## Training

To train the model on the `Fujita` scene, run the following command:

```bash
python run_nerf.py --config configs/plenoptic_config_1im.txt
```
To train the model on the `Rabbit` scene, run the following command:

```bash
python run_nerf.py --config configs/plenoptic_config_rabbit.txt
```

To train the model on the `Fujita Multi-view` scene, run the following command:

```bash
python run_nerf.py --config configs/plenoptic_config_1_11_21.txt
```
