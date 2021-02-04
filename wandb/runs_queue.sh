python -m segmlib --embeddings embeddings/flowers.pth
python -m segmlib --embeddings embeddings/imagenet.pth

python -m segmlib --embeddings embeddings/imagenet.pth --epochs 50 --lr_decay 0.9
