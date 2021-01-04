#!/usr/bin/env bash

mkdir ../weights
cd ../weights

# BigBiGAN-pytorch weights
wget https://www.dropbox.com/s/9w2i45h455k3b4p/BigBiGAN_x1.pth
# background darkening and foreground lightening direction
wget https://www.dropbox.com/s/np74kmkohkbx76t/bg_direction.pth
