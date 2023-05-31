# DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design

This repository is the official implementation of _DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design._


## Dependencies
### Install via Conda and Pip
```bash
conda create -n decompdiff python=3.8
conda activate decompdiff
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For decomposition
conda install -c conda-forge mdtraj
pip install alphaspace2

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Preprocess 
```bash
python scripts/data/preparation/preprocess_subcomplex.py configs/preprocessing/crossdocked.yml
```
We have provided the processed dataset file [here](https://drive.google.com/drive/folders/1z74dKcDKQbwpo8Uf8EJpGi12T4GCD8_Z?usp=share_link).

## Training
To train the model from scratch, you need to download the *.lmdb, *_name2id.pt and split_by_name.pt files and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/train_diffusion_decomp.py configs/training.yml
```

## Sampling
To sample molecules given protein pockets in the test set, you need to download test_index.pkl and test_set.zip files, unzip it and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/sample_diffusion_decomp.py configs/sampling_drift.yml  \
  --outdir $SAMPLE_OUT_DIR -i $DATA_ID --prior_mode {ref_prior, beta_prior}
```
We have provided the trained model checkpoint [here](https://drive.google.com/drive/folders/1JAB5pp25rEM5Wt-i373_rrAyTsLvAACZ?usp=share_link).

If you want to sample molecules with beta priors, you also need to download files in this [directory](https://drive.google.com/drive/folders/1QOQOuDxdKkipYygZU9OIQUXqV9C28J5O?usp=share_link).

## Evaluation
```bash
python scripts/evaluate_mol_from_meta_full.py $SAMPLE_OUT_DIR \
  --docking_mode {none, vina_score, vina_full} \
  --aggregate_meta True --result_path $EVAL_OUT_DIR
```

## Results
- JSD of bond distances

| Bond | liGAN | GraphBP | AR    | Pocket2Mol | TargetDiff | Ours      |
|------|-------|---------|-------|------------|------------|-----------|
| C-C  | 0.601 | 0.368   | 0.609 | 0.496      | 0.369      | **0.359** |
| C=C  | 0.665 | 0.530   | 0.620 | 0.561      | **0.505**  | 0.537     |
| C-N  | 0.634 | 0.456   | 0.474 | 0.416      | 0.363      | **0.344** |
| C=N  | 0.749 | 0.693   | 0.635 | 0.629      | **0.550**  | 0.584     |
| C-O  | 0.656 | 0.467   | 0.492 | 0.454      | 0.421      | **0.376** |
| C=O  | 0.661 | 0.471   | 0.558 | 0.516      | 0.461      | **0.374** |
| C:C  | 0.497 | 0.407   | 0.451 | 0.416      | 0.263      | **0.251** |
| C:N  | 0.638 | 0.689   | 0.552 | 0.487      | **0.235**  | 0.269     |


- JSD of bond angles

| Angle | liGAN | GraphBP | AR    | Pocket2Mol | TargetDiff | Ours      |
|-------|-------|---------|-------|------------|------------|-----------|
| CCC   | 0.598 | 0.424   | 0.340 | 0.323      | 0.328      | **0.314** |
| CCO   | 0.637 | 0.354   | 0.442 | 0.401      | 0.385      | **0.324** |
| CNC   | 0.604 | 0.469   | 0.419 | **0.237**  | 0.367      | 0.297     |
| OPO   | 0.512 | 0.684   | 0.367 | 0.274      | 0.303      | **0.217** |
| NCC   | 0.621 | 0.372   | 0.392 | 0.351      | 0.354      | **0.294** |
| CC=O  | 0.636 | 0.377   | 0.476 | 0.353      | 0.356      | **0.259** |
| COC   | 0.606 | 0.482   | 0.459 | **0.317**  | 0.389      | 0.339     |

- Main results


| Methods    | Vina Score (&darr;) | Vina Min (&darr;) | Vina Dock (&darr;) | High Affinity (&uarr;) | QED (&uarr;) | SA (&uarr;) | Success Rate (&uarr;) |
|------------|---------------------|-------------------|--------------------|------------------------|--------------|-------------|-----------------------|
| Reference  | -6.46               | -6.49             | -7.26              | -                      | 0.47         | 0.74        | 25.0%                 |
| liGAN      | -                   | -                 | -6.20              | 11.1%                  | 0.39         | 0.57        | 3.9%                  |
| GraphBP    | -                   | -                 | -4.70              | 6.7%                   | 0.45         | 0.48        | 0.1%                  |
| AR         | -5.64               | -5.88             | -6.62              | 31.0%                  | 0.50         | 0.63        | 7.1%                  |
| Pocket2Mol | -4.70               | -5.82             | -6.79              | 51.0%                  | 0.57         | 0.75        | 24.4%                 |
| TargetDiff | -6.30               | -6.83             | -7.91              | 59.1%                  | 0.48         | 0.58        | 10.5%                 |
| Ours       | -6.04               | **-7.09**         | **-8.43**          | **71.0%**              | 0.43         | 0.60        | **24.5%**             |


## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](LICENSE).