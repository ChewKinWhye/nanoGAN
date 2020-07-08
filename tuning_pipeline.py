import os
os.system("CUDA_VISIBLE_DEVICES=-1 python fgan.py --epochs 50 --alpha 1 --beta 0 --gamma 1 --pretrain 0 --d_lr 1e-4 --g_lr 2e-4  --latent_dim 50 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename vanilla_gan")

#os.system("CUDA_VISIBLE_DEVICES=1 python fgan.py --epochs 100 --alpha 0.8 --beta 20 --gamma 1 --pretrain 20 --d_lr 1e-4 --g_lr 1e-4  --latent_dim 80 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 002")

#os.system("CUDA_VISIBLE_DEVICES=1 python fgan.py --epochs 50 --alpha 0.8 --beta 20 --gamma 0.8 --pretrain 0 --d_lr 5e-4 --g_lr 5e-4  --latent_dim 50 --data_size 200000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 003")

#os.system("CUDA_VISIBLE_DEVICES=1 python fgan.py --epochs 50 --alpha 1 --beta 0 --gamma 1 --pretrain 15 --d_lr 1e-3 --g_lr 1e-3  --latent_dim 50 --data_size 200000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 004")

print("COMPLETED PIPELINE")

