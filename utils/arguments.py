import argparse


def parse_args():
    parser = argparse.ArgumentParser('Train your Fence GAN')
    parser.add_argument('--data_path', type=str, default='data', help='path to data directory')
    parser.add_argument('--output_filename', type=str, default='gan_model', help='name of the output file')

    # FenceGAN hyper-parameter
    parser.add_argument('--beta', type=float, default=30, help='beta')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--wgan', type=bool, default=True, help='Set True to use wgan')
    # Other hyper-parameters
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--pretrain', type=int, default=15, help='number of pretrain epoch')
    parser.add_argument('--d_l2', type=float, default=0, help='L2 Regularizer for Discriminator')
    parser.add_argument('--d_lr', type=float, default=2e-5, help='learning_rate of discriminator')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='learning rate of generator')
    parser.add_argument('--v_freq', type=int, default=1, help='epoch frequency to evaluate performance')
    parser.add_argument('--seed', type=int, default=0, help='numpy and tensorflow seed')
    parser.add_argument('--latent_dim', type=int, default=100,
                               help='Latent dimension of Gaussian noise input to Generator')
    parser.add_argument('--data_size', type=int, default=200000, help='size of dataset to use')
    parser.add_argument('--label_smoothing', type=bool, default=True, help='To use label smoothing or not')
    args = parser.parse_args()
    return args
