# Codes for Mixup Training for Generative Models to Defend Membership Inference Attacks

## The structure of this repo.

│   models.py
│   readme.md
│
├───attack
│       atk_logan_comember.py
│       atk_logan_comem_against_pargan.py
│       attack_comember.py
│       attack_logan.py
│       LBFGS_pytorch.py
│       wb_ganleaks.py
│       wb_ganleaks_modified.py
│
├───datasetExample
│   └───celeba
│       │   list_attr_celeba.txt
│       │
│       └───img_align_celeba
│               000001.jpg
│               000002.jpg
│
├───drawFigures
│       draw.py
│       histo.py
│
├───trainTargetModel
│       train_gan.py
│       train_gan_mix.py
│       train_pargan.py
│
└───utility
        saveFake.py
        test_downstream.py
        train_downstream.py
        train_downstream_nogen.py

## Instruction

`datasetExample` contains a tiny part of CelebA dataset to show the basic structure of the dataset structure we used. 
To run any codes in this repo, you need to rename `datasetExample` by `dataset` and download the full version from the official website of CelebA. 

`trainTargetModel` contains the scripts to train target DNN models (victims). 

`attack` contains the implementation of membership inference attack algorithms considered in this work.

`utility` contains the scripts to train and test the downstream classifier. Additionally, `saveFake.py` is provided to save fake images generated by target models, which is useful when using the package [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) to calculate Fréchet Inception Distance (FID).

`drawFigures` contains the scripts to visualize privacy result, i.e. the performance of MIA in different settings.