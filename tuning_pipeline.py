import os
#os.system("python main.py --epochs 50 --alpha 0.8 --beta 0 --gamma 1 --pretrain 0 --d_lr 1e-4 --g_lr 1e-4  --latent_dim 50 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 001")

os.system("python main.py --epochs 50 --alpha 0.8 --beta 20 --gamma 1 --pretrain 0 --d_lr 1e-4 --g_lr 1e-4  --latent_dim 50 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 002")

os.system("python main.py --epochs 50 --alpha 0.8 --beta 20 --gamma 0.8 --pretrain 0 --d_lr 1e-4 --g_lr 1e-4  --latent_dim 50 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 003")

#os.system("python main.py --epochs 1 --alpha 1 --beta 0 --gamma 1 --pretrain 15 --d_lr 1e-4 --g_lr 1e-4  --latent_dim 50 --data_size 100000 --data_path /hdd/modifications/ecoli/deepsignal/ --batch_size 512 --output_filename 004")

print("COMPLETED PIPELINE")
