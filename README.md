# Random Normalization Aggregation for Adversarial Defense

## Environment

* torch 1.7.1
* torchvision 0.8.2
* torchattacks 3.2.6

## Training of RNA

* To train a ResNet18 with RNA on CIFAR-10:
```
python train.py --random_norm_training --mixed --network ResNet18 --batch_size 128 --num_group_schedule 0 0 --worker 4 --random_type bn --gn_type gnr --save_dir resnet_c10_RNA
```

## Evaluation of RNA

* To evaluate the performance of ResNet18 with RNA on CIFAR-10:

```
python evaluate.py --random_norm_training --mixed --network ResNet18 --attack_type pgd --batch_size 128 --num_group_schedule 0 0 --worker 4 --random_type bn --gn_type gnr --pretrain ./ckpt/resnet_c10.pth --save_dir resnet_c10_RNA
```

## Pretrained models

Pretrained models are provided in [google-drive](https://drive.google.com/drive/folders/1MusbFsrV7j5UQYF0eqN7zWpghz7ZUM09?usp=sharing), including ResNet18/WideResNet32 on CIFAR-10/100 trained by RNA.

## Acknowledgement

Our codes are modified from [Double-Win Quant](https://github.com/RICE-EIC/Double-Win-Quant).
