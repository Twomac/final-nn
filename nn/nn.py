# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union(int, str)]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    # implemented via Gemini Code Assist
    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise Exception('Non-supported activation function')

        return A_curr, Z_curr

    # implemented via VS code suggestions
    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        # A_prev is the activation matrix of the previous layer.
        # For the first layer, this is the input matrix X.
        # We transpose X to have dimensions (features, batch_size)
        # for vectorized matrix multiplication.
        A_prev = X.T

        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation_curr = layer['activation']

            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation_curr)

            # Cache the previous activation (A_{L-1}) and current linear transform (Z_L)
            cache['A' + str(idx)] = A_prev
            cache['Z' + str(layer_idx)] = Z_curr

            A_prev = A_curr

        output = A_prev # final output is merely last activation matrix

        return output, cache


    # implemented via Gemini Code Assist, 
    # it was confusing because the slides did not contain a dA term explicitly, 
    # but I determined that dZ corresponds to delta from the slides
    # additionally, note that here W_curr is considered to be based on the layer
    # that W connects to, therefore W_curr for the first backpropgation is actually
    # associated with the final layer in the neural network, this was also different
    # from how the slides labeled the weights
    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Get the number of examples in the batch
        m = A_prev.shape[1]

        # Step 1: Calculate dZ_curr
        if activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise Exception("Non-supported activation function")

        # Step 2: Calculate dW_curr and db_curr
        # dW = (1/m) * dZ . A_prev^T
        dW_curr = (1/m) * np.dot(dZ_curr, A_prev.T)
        # db = (1/m) * sum(dZ) across the batch dimension
        db_curr = (1/m) * np.sum(dZ_curr, axis=1, keepdims=True)

        # Step 3: Calculate dA_prev
        # dA_prev = W^T . dZ
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr


    # implemented via VS code suggestions + Gemini Code Assist
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}

        # 1. Calculate the initial gradient of the loss function w.r.t the final activation
        if self._loss_func == "binary_cross_entropy":
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == "mean_squared_error":
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise Exception(f"Loss function {self._loss_func} not supported")
        
        for idx in reversed(range(len(self.arch))):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(idx)]
            activation_curr = self.arch[idx]['activation']

            dA_curr, dW_curr, db_curr = self._single_backprop(
                W_curr,
                b_curr,
                Z_curr,
                A_prev,
                dA_curr,
                activation_curr
            )

            # Store the gradients in a dictionary for use in parameter update step
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
            
        return grad_dict

    # 
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, X_train.shape[0], self._batch_size):
                X_batch = X_train_shuffled[i:i + self._batch_size]
                y_batch = y_train_shuffled[i:i + self._batch_size]
                
                # Reshape y_batch if 1D to ensure correct broadcasting (N, 1)
                if y_batch.ndim == 1:
                    y_batch = y_batch.reshape(-1, 1)
                
                # Forward pass
                y_hat, cache = self.forward(X_batch)
                
                # Compute loss (transpose y to match y_hat shape: features x batch)
                if self._loss_func == "binary_cross_entropy":
                    loss = self._binary_cross_entropy(y_batch.T, y_hat)
                elif self._loss_func == "mean_squared_error":
                    loss = self._mean_squared_error(y_batch.T, y_hat)
                
                epoch_loss += loss
                num_batches += 1

                # Backpropagation and update
                grad_dict = self.backprop(y_batch.T, y_hat, cache)
                self._update_params(grad_dict)

            per_epoch_loss_train.append(epoch_loss / num_batches)

            # Validation loss
            y_val_reshaped = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
            y_hat_val, _ = self.forward(X_val)
            if self._loss_func == "binary_cross_entropy":
                per_epoch_loss_val.append(self._binary_cross_entropy(y_val_reshaped.T, y_hat_val))
            elif self._loss_func == "mean_squared_error":
                per_epoch_loss_val.append(self._mean_squared_error(y_val_reshaped.T, y_hat_val))

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat.T

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = dA * self._sigmoid(Z) * (1 - self._sigmoid(Z)) # derivative has closed form in terms of sigmoid function
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0, Z)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        dZ[Z <= 0] = 0  # When z <= 0, you should set dz to 0 as well.
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        m = y_hat.shape[1]
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return float(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        dA = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        m = y_hat.shape[1]
        loss = (1/m) * np.sum(np.square(y - y_hat))
        return float(loss)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = 2 * (y_hat - y)
        return dA