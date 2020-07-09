import os
# Command for vanilla GAN
os.system("CUDA_VISIBLE_DEVICES=-1 python fgan.py --epochs 300 --alpha 1 --beta 0 --gamma 1 --pretrain 0 --d_lr 1e-4 --g_lr 2e-4  --latent_dim 50 --data_size 1500000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename vanilla_gan")

# Command for supervised learning
# os.system("CUDA_VISIBLE_DEVICES=-1 python supervised.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 1500000")

print("COMPLETED PIPELINE")

