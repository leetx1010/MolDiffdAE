Code for "Disentangled Autoencoding Equivariant Diffusion Model for Controlled Generation of 3D Molecules"
======

***Training***

python3 scripts/train_drug3d_ae.py --config configs/train/train_MolDiffAE_${mode}.yml --emb_dim ${emb_dim} --wass_weight ${wass_weight} --batch_size ${batch_size} --logdir ${savedir}

We provide a small sample dataset with 2000 molecules (geom_drug_test.zip). The full dataset can be downloaded from [this repo](https://github.com/pengxingang/MolDiff)


***WIP***

1. Data for property manipulation (property_prediction_data_val/test.pkl)
2. Script for data preprocessing
