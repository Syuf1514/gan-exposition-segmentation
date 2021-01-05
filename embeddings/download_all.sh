#!/usr/bin/env bash


# CUB 2010-2011 train embeddings
  wget -O cub.npy https://www.dropbox.com/s/qgacaoo7urh35j9/BigBiGAN_CUB_train_z.npy -q --show-progress

# DUTS
wget -O duts.npy https://www.dropbox.com/s/b77in0vuc8jy1yf/BigBiGAN_DUTS-TR_z.npy -q --show-progress

# Flowers
wget -O flowers.npy https://www.dropbox.com/s/fmw2g54xqli59ck/BigBiGAN_Flowers_train_z.npy -q --show-progress

# ImageNet
wget -O imagenet.npy https://www.dropbox.com/s/y5qr9wwtop7ot6b/BigBiGAN_ImageNet_z.npy -q --show-progress
