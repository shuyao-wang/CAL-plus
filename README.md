# CAL-plus
We provide a detailed code for CAL-plus.

## Installations
Main packages: PyTorch, Pytorch Geometric, OGB.
```
pytorch==1.10.1
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
ogb==1.3.4
```

## Preparations
Please download the graph OOD datasets and OGB datasets as described in the original paper. 
Create a folder ```dataset```, and then put the datasets into ```dataset```. Then modify the path by specifying ```--data_dir your/path/dataset```.


## Commands
 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments.
 To run the code on HIV, please use the following command:
 ```
 CUDA_VISIBLE_DEVICES=$GPU nohup  python -u  train_real.py \
 --dataset hiv \
 --domain size \
 --beta=1 \
 --lr=0.001 \
 --min_lr 1e-6 \
 --weight_decay 0.001 \
 --hidden 300 \
 --epoch 100 \
 --batch_size 256 \
 --trails 10 \
 --memory True \
 --prototype True \
 --me_batch_n 3 >> ./log/hiv_size.log 2>&1 &

 ```


 To run the code on Molbbbp, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU nohup  python -u ./molbbbp/train_real.py \
 --dataset ogbg-molbbbp \
 --domain size \
 --beta 0.05 \
 --co 0.1 \
 --c 0.1 \
 --o 1 \
 --lr=0.001 \
 --min_lr 1e-8 \
 --weight_decay 5e-6 \
 --hidden 64 \
 --layers 2 \
 --epochs 100 \
 --pretrain 50 \
 --batch_size 128 \
 --trails 10 \
 --memory True \
 --prototype True \
 --me_batch_n 1 >> ./log/molbbbp_size.log 2>&1 &
 ```

To run the code on Motif, please use the following command:
 ```
 CUDA_VISIBLE_DEVICES=$GPU nohup  python -u train_real.py \
 --dataset motif \
 --domain size \
 --beta 0.5 \
 --lr 0.001 \
 --min_lr 1e-3 \
 --weight_decay 0 \
 --hidden 128 \
 --epoch 400 \
 --batch_size 256 \
 --trails 10 \
 --memory True \
 --prototype True \
 --me_batch_n 5 >> ./log/motif_size.log 2>&1 &
 ```

 To run the code on CMNIST, please use the following command:
 ```
 CUDA_VISIBLE_DEVICES=$GPU nohup  python -u train_real.py \
 --dataset cmnist \
 --domain color \
 --beta 1 \
 --lr=0.001 \
 --layers=5 \
 --min_lr 1e-6 \
 --weight_decay 0 \
 --hidden 128 \
 --epoch 200 \
 --batch_size 256 \
 --memory True \
 --prototype True \
 --me_batch_n 6 >> ./log/cmnist_color.log 2>&1 &


 ```

 To run the code on SYN, please use the following command:

```
lr=0.002
min=5e-6
b=0.5

CUDA_VISIBLE_DEVICES=$GPU python -u main.py --bias $b --lr $lr --min_lr $min --model CausalGIN \
--beta 0.05 \
--memory True \
--prototype True \
--me_batch_n 3

```

