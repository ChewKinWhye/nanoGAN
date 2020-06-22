from utils.evaluate import plot_prc
import os
import json


def create_directories(args):
    result_path = f'./results/{args.output_filename}/'
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def save_model(args, D):
    create_directories(args)
    result_path = f'./results/{args.output_filename}/'
    D.save(f'{result_path}/discriminator.h5')


def save_results(args, results):
    create_directories(args)
    result_path = f'./results/{args.output_filename}/'

    result_json = ["best_au_prc", round(results[0], 3)]

    with open(f'{result_path}/result.json', 'w+') as outfile:
        json.dump(result_json, outfile, indent=4)

    plt = plot_prc(results[1], results[2])
    plt.savefig(f"{result_path}/prc.png")
