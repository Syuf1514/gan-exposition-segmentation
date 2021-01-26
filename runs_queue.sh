python -m segmlib --train_embeddings embeddings/cub_train.pth --valid_embeddings embeddings/cub_valid.pth --epochs 1
python -m segmlib --train_embeddings embeddings/flowers_train.pth --valid_embeddings embeddings/flowers_valid.pth --epochs 1
python -m segmlib --train_embeddings embeddings/imagenet_train.pth --valid_embeddings embeddings/imagenet_valid.pth --epochs 1

python -m segmlib --train_embeddings embeddings/cub_train.pth --valid_embeddings embeddings/cub_valid.pth --epochs 5
python -m segmlib --train_embeddings embeddings/flowers_train.pth --valid_embeddings embeddings/flowers_valid.pth --epochs 5
python -m segmlib --train_embeddings embeddings/imagenet_train.pth --valid_embeddings embeddings/imagenet_valid.pth --epochs 5

python -m segmlib --train_embeddings embeddings/cub_train.pth --valid_embeddings embeddings/cub_valid.pth --epochs 10
python -m segmlib --train_embeddings embeddings/flowers_train.pth --valid_embeddings embeddings/flowers_valid.pth --epochs 10
python -m segmlib --train_embeddings embeddings/imagenet_train.pth --valid_embeddings embeddings/imagenet_valid.pth --epochs 10
