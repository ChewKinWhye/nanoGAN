## Prerequisites
1. Linux OS
2. Python 3
3. CUDA 

## Installation
1. Clone repository
    ```
    git clone https://github.com/ChewKinWhye/nanoGAN.git
    ```
2. Installing tensorflow or tensorflow-gpu by following instruction [here](https://www.tensorflow.org/install/pip).

3. Installing necessary libraries
    ```
    pip3 install -r requirements.txt
    ```

## Anomaly Detection
Check results and plots under `results` folder
### Ecoli data set
```
python main.py --epochs 150 --alpha 0.1 --beta 30 --gamma 0.1 --pretrain 15 --d_lr 1e-5 --g_lr 2e-5  --latent_dim 200 --data_size 100000
```

### More training option
Enter `python3 main.py -h` for more training options
```
    usage: Train your Fence GAN [-h] [--dataset DATASET] [--ano_class ANO_CLASS]
                                [--epochs EPOCHS] [--beta BETA] [--gamma GAMMA]
                                [--alpha ALPHA] [--batch_size BATCH_SIZE]
                                [--pretrain PRETRAIN] [--d_l2 D_L2] [--d_lr D_LR]
                                [--g_lr G_LR] [--v_freq V_FREQ] [--seed SEED]
                                [--evaluation EVALUATION]
                                [--latent_dim LATENT_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset         mnist | cifar10
      --ano_class       1 anomaly class
      --epochs          number of epochs to train
      --beta            beta
      --gamma           gamma
      --alpha           alpha
      --batch_size 
      --pretrain        number of pretrain epoch
      --d_l2            L2 Regularizer for Discriminator
      --d_lr            learning_rate of discriminator
      --g_lr            learning rate of generator
      --v_freq          epoch frequency to evaluate performance
      --seed            numpy and tensorflow seed
      --evaluation      'auprc' or 'auroc'
      --latent_dim      Latent dimension of Gaussian noise input to Generator
  ```
  
