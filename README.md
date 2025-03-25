Code for "Disentangled Autoencoding Equivariant Diffusion Model for Controlled Generation of 3D Molecules"
======

***Training***

python3 scripts/train_drug3d_ae.py --config configs/train/train_MolDiffAE_${mode}.yml --emb_dim ${emb_dim} --wass_weight ${wass_weight} --batch_size ${batch_size} --logdir ${savedir}

We provide a small sample dataset with 2000 molecules (data/geom_drug_test.zip). The full dataset can be downloaded from [this repo](https://github.com/pengxingang/MolDiff)

***Guided sampling***

python3 scripts/sample_drug3d_ae_ddim_template.py --config configs/train/train_MolDiffAE_${mode}.yml --name ${dataset} --logdir ${logdir} --ckpt ${ckpt}  --noise ${noise}

noise: "deterministic" or "random" $x_T$

Sample outputs with DDIM (requires rdkit):

- output/template-drug3d_ddim-999-deterministic_test.pkl: source embedding + deterministic $x_T$ (perfect reconstruction expected)

- output/template-drug3d_ddim-999-random_test.pkl: source embedding + random $x_T$ (neighborhood search of source) 

***WIP***

1. Data for property manipulation (property_prediction_data_val/test.pkl)
2. Script for data preprocessing
3. Property manipulation demo and results
