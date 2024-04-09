# OFUNet
Github from the SPIE 2024 Medical Imaging paper "Exploring optical flow inclusion into nnU-Net framework for surgical instrument segmentation". 
[(SPIE poster+paper)](https://doi.org/10.1117/12.3006855) and 
[(Link to ArXiv paper)](https://arxiv.org/abs/2403.10216)


This work made some minimal modifications to the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework to include optical flow (OF) as an additional input.

![Workflow_git_light](https://github.com/MarcosFdzRdz/OFUNet/assets/100223846/efe05cf4-b5c3-442c-a48f-21da891edf7d)


## Modifications tutorial
For the inclusion of the OF into the nnUNet framework, some of the augmentations were disabled.
The file `nnUNetTrainer_OF.py` is the modified trainer which includes these changes and was used for training all the models of our work.

In order to use it in the nnUNet_v2, it must be installed in your computer following [the instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).
Once installed, the file `nnUNetTrainer_OF.py` must be added to the nnUNetTrainer folder (example: /home/user/nnUNet/nnunetv2/training/nnUNetTrainer).

Then follow the regular [usage instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) but calling the modified trainer adding `-tr nnUNetTrainer_OF` at the end of the training and inference commands `nnUNetv2_train` and `nnUNetv2_predict`.

## Video with results 
Video showing results from the nnUNet model without OF (RGB), our nnUNet model with additional OF input (RGBof), the original ground truth from the dataset (GT), and the OF map added as an input for the RGBof (OF).

https://github.com/MarcosFdzRdz/OFUNet/assets/100223846/17e86d4b-e520-43a2-b5d6-e172bfbc58ad
