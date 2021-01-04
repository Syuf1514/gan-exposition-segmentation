#!/usr/bin/env bash

mkdir ../embeddings
cd ../embeddings

# CUB 2010-2011 train embeddings
wget https://www.dropbox.com/s/qgacaoo7urh35j9/BigBiGAN_CUB_train_z.npy
# DUTS
wget https://www.dropbox.com/s/b77in0vuc8jy1yf/BigBiGAN_DUTS-TR_z.npy
# Flowers
wget https://www.dropbox.com/s/fmw2g54xqli59ck/BigBiGAN_Flowers_train_z.npy
# ImageNet
wget https://www.dropbox.com/s/y5qr9wwtop7ot6b/BigBiGAN_ImageNet_z.npy
