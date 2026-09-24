import copy
import numpy as np
import sys
sys.path += ['layers']
from loss_crossentropy import loss_crossentropy

######################################################
from inference import inference
from calc_gradient import calc_gradient
from update_weights import update_weights
######################################################


def train(
        model,
        input,
        label,
        val_input,
        val_label,
        params,
        numIters,
        numItersPerEval):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        val_input: [any_dimensions] x [val_num_inputs]
        val_label: [val_num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
        numItersPerEval: Number of iterations before computing loss / accuracy on validation set
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    # Optimization method, default SGD
    optimization = params.get("optimization", "SGD")
    # momentum friction
    rho = params.get("rho", 0.9)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features
    # like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd}

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))

    num_val = val_input.shape[-1]
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    best_accuracy = 0
    best_model = {}
    num_layers = len(model["layers"])
    velocities = [None, ] * num_layers

    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
        pass

    ########
    return model, loss, val_losses, val_accuracies, train_losses, train_accuracies
