Code for "Disentangled Autoencoding Equivariant Diffusion Model for Controlled Generation of 3D Molecules"
======

***Environment***

Pleae first install conda

conda env create --name moldiffdae --file env.yml

conda activate moldiffdae

conda install pip

Use pip to install torch (cuda-toolkit version should match system cuda version; higher versions of torch may also work):

      - torch==2.0.1+cu118
      
      - torch-cluster==1.6.3+pt20cu118
      
      - torch-geometric==2.3.1
      
      - torch-scatter==2.1.2+pt20cu118
      
      - torch-sparse==0.6.18+pt20cu118
      
      - torch-spline-conv==1.2.2+pt20cu118
      
      - torchmetrics==0.11.4
      
      - cuda-toolkit=11.8.0=0
      
      - cuda-tools=11.8.0=0

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
