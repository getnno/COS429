"""
Basic script to create a new network model.
The model presented here is meaningless, but it shows how to properly
call init_model and init_layers for the various layer types.
"""

from train import train
from data_utils import get_digits
from loss_euclidean import loss_euclidean
from inference import inference
from init_model import init_model
from init_layers import init_layers
import numpy as np
import sys
sys.path += ['layers']


def main():
    l = [init_layers('conv', {'filter_size': 3,
                              'filter_depth': 1,
                              'num_filters': 4}),
         # init_layers('pool', {'filter_size': 2,
         #                     'stride': 2}),
         init_layers('relu', {}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 144,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [8, 8, 1], 10, True)

    # Example calls you might make for inference:
    inp = np.random.rand(8, 8, 1, 3)    # Dummy input
    output, _ = inference(model, inp)

    X_train, y_train, X_test, y_test = get_digits()

    params = {
        'learning_rate': 0.7,
        'batch_size': 128,
        'save_file': "model.npz"
    }
    model, loss, val_losses, val_accuracies, train_losses, train_accuracies = train(
        model, X_train, y_train, X_test, y_test, params, 5, 1)


if __name__ == '__main__':
    main()
