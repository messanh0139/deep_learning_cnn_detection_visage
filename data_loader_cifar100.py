"""
Module de chargement et de préparation des données CIFAR-100
"""

import numpy as np
from keras.datasets import cifar100

def load_cifar100():
    """
    Charge le dataset CIFAR-100
    
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
    """
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (X_train, y_train), (X_test, y_test)
