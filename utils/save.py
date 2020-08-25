import os
import json


def create_directories(args):
    result_path = f'./results/{args.output_filename}/'
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def save_gan_model(args, model):
    create_directories(args)
    result_path = f'./results/{args.output_filename}/'
    model.save(f'{result_path}/discriminator.h5')


def save_results(args, results):
    create_directories(args)
    result_path = f'./results/{args.output_filename}'

    result_json = {
        "Best au_prc": round(results[0], 3),
        "Best au_roc": round(results[3], 3),
        "Best accuracy": round(results[6], 3),
        "Best F-Measure": round(results[7], 3)}

    with open(f'{result_path}/result.json', 'w+') as outfile:
        json.dump(result_json, outfile, indent=4)


def save_vae_model_dna(args, encoder, predictor, min_values, max_values):
    result_path = f'./results/{args.output_filename}'
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    encoder.save(f'{result_path}/encoder.h5')
    predictor.save(f'{result_path}/predictor.h5')

    with open(f'{result_path}/min_values.json', 'w') as f:
        json.dump(min_values, f, indent=2)
    with open(f'{result_path}/max_values.json', 'w') as f:
        json.dump(max_values, f, indent=2)


def save_vae_model_rna(args, encoder, predictor):
    result_path = f'./results/{args.output_filename}'
    if not os.path.exists('./results'):
        os.makedirs('/.results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    encoder.save(f'{result_path}/encoder.h5')
    predictor.save(f'{result_path}/predictor.h5')
