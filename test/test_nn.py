import numpy as np
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

def test_single_forward():
    # Initialize a simple network to access methods
    nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error')
    
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0], [1.0]])
    A_prev = np.array([[1.0], [1.0]])
    
    # Expected Z = W.A + b = [[1*1 + 2*1 + 1], [3*1 + 4*1 + 1]] = [[4], [8]]
    # Expected A (relu) = [[4], [8]]
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'relu')
    
    assert np.array_equal(A_curr, np.array([[4.0], [8.0]]))
    assert np.array_equal(Z_curr, np.array([[4.0], [8.0]]))

def test_forward():
    nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=42, batch_size=2, epochs=1, loss_function='mean_squared_error')
    
    # Batch size 2, features 2
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    output, cache = nn.forward(X)
    
    # Output should be (output_dim, batch_size) -> (1, 2)
    assert output.shape == (1, 2)
    assert 'Z1' in cache
    assert 'A1' in cache
    assert 'Z2' in cache

def test_single_backprop():
    nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error')

    W = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([[0.0], [0.0]])
    Z = np.array([[1.0, 1.0], [1.0, 1.0]])
    A_prev = np.array([[1.0, 1.0], [1.0, 1.0]])
    dA_curr = np.array([[1.0, 1.0], [1.0, 1.0]])

    dA_prev, dW, db = nn._single_backprop(W, b, Z, A_prev, dA_curr, 'relu')

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

def test_predict():
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error')
    X = np.array([[1.0, 1.0], [0.0, 0.0]])
    
    # Predict returns transpose of forward output: (Batch, Output)
    preds = nn.predict(X)
    assert preds.shape == (2, 1)

def test_binary_cross_entropy():
    nn = NeuralNetwork([], 0.1, 42, 1, 1, 'binary_cross_entropy')
    y = np.array([[1, 0]])
    y_hat = np.array([[0.9, 0.1]])
    
    # Loss = -1/2 * (log(0.9) + log(0.9)) = -log(0.9) approx 0.10536
    loss = nn._binary_cross_entropy(y, y_hat)
    assert np.isclose(loss, -np.log(0.9))

def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork([], 0.1, 42, 1, 1, 'binary_cross_entropy')
    y = np.array([[1]])
    y_hat = np.array([[0.5]])
    
    # dA = - (y/y_hat - (1-y)/(1-y_hat)) = -(2 - 0) = -2
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    assert np.isclose(dA, -2.0)

def test_mean_squared_error():
    nn = NeuralNetwork([], 0.1, 42, 1, 1, 'mean_squared_error')
    y = np.array([[1, 0]])
    y_hat = np.array([[0.5, 0.5]])
    
    # MSE = 1/2 * ((0.5)^2 + (-0.5)^2) = 1/2 * (0.25 + 0.25) = 0.25
    loss = nn._mean_squared_error(y, y_hat)
    assert np.isclose(loss, 0.25)

def test_mean_squared_error_backprop():
    nn = NeuralNetwork([], 0.1, 42, 1, 1, 'mean_squared_error')
    y = np.array([[1]])
    y_hat = np.array([[0.5]])
    
    # dA = 2 * (y_hat - y) = 2 * (-0.5) = -1
    dA = nn._mean_squared_error_backprop(y, y_hat)
    assert np.isclose(dA, -1.0)

def test_sample_seqs():
    seqs = ["A", "B", "C"]
    labels = [True, False, False] # 1 Pos, 2 Neg
    
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    
    # Should upsample positive to match negative count (2)
    # Total length should be 4
    assert len(sampled_seqs) == 4
    assert sampled_labels.count(True) == 2
    assert sampled_labels.count(False) == 2

def test_one_hot_encode_seqs():
    seqs = ["AT"]
    # A=[1,0,0,0], T=[0,1,0,0] -> AT=[1,0,0,0, 0,1,0,0]
    expected = np.array([[1, 0, 0, 0, 0, 1, 0, 0]])
    result = one_hot_encode_seqs(seqs)
    assert np.array_equal(result, expected)