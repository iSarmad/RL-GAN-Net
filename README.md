# RL-GAN-Net
Official Repository of CVPR 2019 Paper : RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion


https://arxiv.org/abs/1904.12304


Requirements:

The packages in my Conda Environment are listed in Requirement_Conda.txt and Requirements_pip.txt files. Only install the ones needed or you can clone the whole environment. 

Steps
* Visualize each training and testing step by using visdom.

1. Download data from https://github.com/optas/latent_3d_points.
2. Process Data with Processdata2.m to get incomplete point cloud
3. Train the autoencoder using main.py and save the model
4. Generate GFV  using pretrained AE using GFV.py and store data
5. Train GAN on the generated GFV data by by going into the GAN folder (trainer.py) and save model
6. Train RL by using pre-trained GAN and AE by running trainRL.py
7. Test with Incomplete data by running testRL.py

Credits:

1. https://github.com/optas/latent_3d_points
2. https://github.com/heykeetae/Self-Attention-GAN
3. https://github.com/lijx10/SO-Net (for chamfer distance)
4. https://github.com/sfujim/TD3



If you use this work for your projects, please take the time to cite our CVPR paper:

```
@InProceedings{Sarmad_2019_CVPR,
author = {Sarmad, Muhammad and Lee, Hyunjoo Jenny and Kim, Young Min},
title = {RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
