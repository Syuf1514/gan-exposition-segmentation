#!/usr/bin/env bash


# BigBiGAN-pytorch weights
wget -O bigbigan.pth https://www.dropbox.com/s/9w2i45h455k3b4p/BigBiGAN_x1.pth -q --show-progress

# background darkening and foreground lightening direction
wget -O direction.pth https://www.dropbox.com/s/np74kmkohkbx76t/bg_direction.pth -q --show-progress
