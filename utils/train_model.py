import os
import json
import random
from tqdm import trange
from collections import OrderedDict

import numpy as np

import keras.backend as K
import matplotlib.pyplot as plt

from utils.model import *
from utils.visualize import show_images, compute_au, histogram


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def noise_data(n_samples, latent_dim):
        return np.random.normal(0,1,[n_samples,latent_dim])

    
def D_data(n_samples,G,mode,x_train,latent_dim):
    #Feeding training data for normal case
    if mode == 'normal':
        sample_list = random.sample(list(range(np.shape(x_train)[0])), n_samples)
        x_normal = x_train[sample_list,...]
        y1 = np.ones(n_samples)
        
        return x_normal, y1
    
    #Feeding training data for generated case    
    if mode == 'gen':
        noise = noise_data(n_samples, latent_dim)
        x_gen = G.predict(noise)
        y0 = np.zeros(n_samples)
        
        return x_gen, y0
    
def pretrain(args, G ,D, x_train):
    ###Pretrain discriminator
    ###Generator is not trained
    print("===== Start of Pretraining =====")
    batch_size = args.batch_size
    pretrain_epoch = args.pretrain
    latent_dim = args.latent_dim
    for e in range(pretrain_epoch):
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Pretrain_Epoch {}'.format(e+1)) as t:
            for step in t:                
                loss = 0
                set_trainability(D, True)
                K.set_value(gamma, [1])
                x,y = D_data(batch_size,G,'normal',x_train, latent_dim)
                loss += D.train_on_batch(x, y)
                
                set_trainability(D, True)
                K.set_value(gamma, [args.gamma])
                x,y = D_data(batch_size,G,'gen',x_train, latent_dim)
                loss += D.train_on_batch(x,y)
                
                t.set_postfix(D_loss=loss/2)
        print("\tDisc. Loss: {:.3f}".format(loss/2))
    print("===== End of Pretraining =====")
        
        
def train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val):
    ###Adversarial Training
    epochs = args.epochs
    batch_size = args.batch_size
    v_freq= args.v_freq
    ano_class = args.ano_class
    latent_dim = args.latent_dim
    evaluation = args.evaluation

    if not os.path.exists('./result/{}/'.format(args.dataset)):
        os.makedirs('./result/{}/'.format(args.dataset))
    result_path = './result/{}/{}'.format(args.dataset,len(os.listdir('./result/{}/'.format(args.dataset))))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    d_loss = []
    g_loss = []
    best_val = 0
    best_test = 0
    best_precision = []
    best_recall = []
    best_fpr = []
    best_tpr = []
    best_epoch = 0
    print('===== Start of Adversarial Training =====')
    for epoch in range(epochs):
        try:
            with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                for step in t:
                    ###Train Discriminator
                    loss_temp = []
                    
                    set_trainability(D, True)
                    K.set_value(gamma, [1])
                    x,y = D_data(batch_size,G,'normal',x_train, latent_dim)
                    loss_temp.append(D.train_on_batch(x, y))
                    
                    set_trainability(D, True)
                    K.set_value(gamma, [args.gamma])
                    x,y = D_data(batch_size,G,'gen',x_train, latent_dim)
                    loss_temp.append(D.train_on_batch(x,y))
                    
                    d_loss.append(sum(loss_temp)/len(loss_temp))
                    
                    ###Train Generator
                    set_trainability(D, False)
                    x = noise_data(batch_size, latent_dim)
                    y = np.zeros(batch_size)
                    y[:] = args.alpha
                    g_loss.append(GAN.train_on_batch(x,y))
                    
                    t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1])
        except KeyboardInterrupt: #hit control-C to exit and save video there
            break    
        
        if (epoch + 1) % v_freq == 0:
            val_prc, test_prc, precision, recall, fpr, tpr, val_roc = compute_au(D, G, GAN, x_val, y_val, x_test, y_test, evaluation)
            f = open('{}/logs.txt'.format(result_path),'a+')
            f.write('\nEpoch: {}\n\t Val_{}: {:.3f} \n\t Test_{}: {:.3f}'.format(epoch+1, evaluation, val_prc, evaluation, test_prc))
            f.close()
            
            if val_roc > best_val:
                best_val = val_roc
                best_test = test_prc
                best_precision = precision
                best_recall = recall
                best_fpr = fpr
                best_tpr = tpr
                best_epoch = epoch

                histogram(G, D, GAN, x_test, y_test, result_path, latent_dim)
                # show_images(G.predict(noise_data(25, latent_dim)),result_path)
                
                G.save('{}/gen_anoclass_{}.h5'.format(result_path,ano_class))
                D.save('{}/dis_anoclass_{}.h5'.format(result_path,ano_class))
                
            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}\n\t{}: {:.3f}".format(g_loss[-1], d_loss[-1], evaluation, val_prc))
            print(val_roc)
        else:
            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}".format(g_loss[-1], d_loss[-1]))

    plt.plot(best_recall, best_precision, 'r-')
    plt.plot(best_fpr, best_tpr, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.savefig(f"./experiment_001_epoch_{best_epoch}_lstm.png")

    print('===== End of Adversarial Training =====')
    print('Dataset: {}| Anomalous class: {}| Best test {}: {}'.format(args.dataset, ano_class, evaluation, round(best_test,3)))
    
    #Saving result in result.json file    
    result =[("best_test_{}".format(evaluation),round(best_test,3)),("best_val_{}".format(evaluation),round(best_val,3))]
    result_dict = OrderedDict(result)
    with open('{}/result.json'.format(result_path),'w+') as outfile:
        json.dump(result_dict, outfile, indent=4)