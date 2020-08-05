import numpy as np
from tqdm import trange

from utils.model import K, gamma, set_trainability
from utils.evaluate import compute_metrics_standardized
from utils.data import D_data, noise_data
from utils.save import save_model


def pre_train(args, generator, discriminator, x_train):
    # Pre-train discriminator, Generator is not trained
    print("===== Start of Pre-training =====")
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    for e in range(args.pretrain):
        loss = 0
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Pre-train_Epoch {}'.format(e+1)) as t:
            for _ in t:
                loss = 0
                set_trainability(discriminator, True)
                K.set_value(gamma, [1])
                x, y = D_data(batch_size, generator, 'normal', x_train, latent_dim)
                loss += discriminator.train_on_batch(x, y)
                set_trainability(discriminator, True)
                K.set_value(gamma, [args.gamma])
                x, y = D_data(batch_size, generator, 'gen', x_train, latent_dim)
                loss += discriminator.train_on_batch(x, y)
                t.set_postfix(D_loss=loss/2)
        print(f"\tDisc. Loss: {loss/2:.3f}")
    print("===== End of Pre-training =====")
        
        
def train(args, generator, discriminator, GAN, x_train, x_test, y_test, x_val, y_val):
    # Adversarial Training
    epochs = args.epochs
    batch_size = args.batch_size
    v_freq = args.v_freq
    latent_dim = args.latent_dim

    d_loss, g_loss, best_cm = [], [], []
    best_au_roc_val, best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc = 0, 0, 0, 0, 0, 0

    print('===== Start of Adversarial Training =====')
    for epoch in range(epochs):
        try:
            with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                for _ in t:
                    # Train Discriminator
                    loss_temp = []
                    set_trainability(discriminator, True)
                    K.set_value(gamma, [1])
                    x, y = D_data(int(batch_size/2), generator, 'normal', x_train, latent_dim)
                    loss_temp.append(discriminator.train_on_batch(x, y))
                    set_trainability(discriminator, True)
                    K.set_value(gamma, [args.gamma])
                    x, y = D_data(int(batch_size), generator, 'gen', x_train, latent_dim)
                    loss_temp.append(discriminator.train_on_batch(x, y))
                    d_loss.append(sum(loss_temp)/len(loss_temp))
                    
                    # Train Generator
                    set_trainability(discriminator, False)
                    x = noise_data(batch_size, latent_dim)
                    y = np.ones(batch_size)
                    y[:] = args.alpha
                    g_loss.append(GAN.train_on_batch(x, y))
                    t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1])
        except KeyboardInterrupt:
            # hit control-C to exit
            break    
        
        if (epoch + 1) % v_freq == 0:
            # Check for the best validation results
            y_predicted = 1 - np.squeeze(discriminator.predict_on_batch(x_val))
            x = noise_data(batch_size, latent_dim)
            print(generator.predict_on_batch(x)[0:5, 24:44])
            accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(y_predicted, y_val)

            if au_roc_val > best_au_roc_val:
                best_au_roc_val = au_roc_val
                # Save the best test results
                y_predicted = 1 - np.squeeze(discriminator.predict_on_batch(x_test))
                best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc, best_cm = compute_metrics_standardized(y_predicted, y_test)
                save_model(args, discriminator)
                
            print(f"\tAccuracy    : {accuracy_val:.3f}")
            print(f"\tSensitivity : {sensitivity_val:.3f}")
            print(f"\tSpecificity : {specificity_val:.3f}")
            print(f"\tPrecision   : {precision_val:.3f}")
            print(f"\tAUC         : {au_roc_val:.3f}")
            print(f"{cm_val}")
        print(f"\tGen. Loss: {g_loss[-1]:.3f}\n\tDisc. Loss: {d_loss[-1]:.3f}")

    print('===== End of Adversarial Training =====')
    print(f"\tBest accuracy    : {best_accuracy:.3f}")
    print(f"\tBest sensitivity : {best_sensitivity:.3f}")
    print(f"\tBest specificity : {best_specificity:.3f}")
    print(f"\tBest precision   : {best_precision:.3f}")
    print(f"\tBest AUC         : {best_au_roc:.3f}") 
    print(f"{best_cm}")
    results = (best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc)
    return results
