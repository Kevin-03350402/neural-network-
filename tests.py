import unittest
import numpy as np
import pickle as pk
from numpy.testing import assert_allclose

from neuralnet import *

TOLERANCE = 1e-4

with open("unittest_data.pk", "rb") as f:
    data = pk.load(f)

# to run one test: python -m unittest tests.TestLinear
# to run all tests: python -m unittest tests


class TestLinear(unittest.TestCase):
    def test_1(self):
        T1, T2 = data["linear_forward"]
        w, X, soln = T1
        a = linear(X, w)
        assert_allclose(np.squeeze(a), soln)
    
    def test_2(self):
        T1, T2 = data["linear_forward"]
        w, X, soln = T2
        b = linear(X, w)
        assert_allclose(np.squeeze(b), soln)


class TestSigmoid(unittest.TestCase):
    def test_1(self):
        T1, T2 = data["sigmoid_forward"]
        a, soln = T1
        z = sigmoid(a)
        assert_allclose(np.squeeze(z), soln)
    
    def test_2(self):
        T1, T2 = data["sigmoid_forward"]
        a, soln = T2
        z = sigmoid(a)
        assert_allclose(np.squeeze(z), soln)


class TestSoftmax(unittest.TestCase):
    def test_1(self):
        T1, T2 = data["softmax_forward"]
        z, soln = T1
        yh = softmax(z)
        assert_allclose(np.squeeze(yh), soln)
    
    def test_2(self):
        T1, T2 = data["softmax_forward"]
        z, soln = T2
        yh = softmax(z)
        assert_allclose(np.squeeze(yh), soln)


class TestCrossEntropy(unittest.TestCase):
    def test(self):
        T = data["ce_forward"]
        yh, y, soln = T
        loss = cross_entropy(y, yh)
        assert_allclose(np.squeeze(loss), soln)


class TestDLinear(unittest.TestCase):
    def test(self):
        T = data["linear_backward"]
        X, w, xsoln, wsoln = T
        dX, dw = d_linear(X, w)
        assert_allclose(np.squeeze(dX), xsoln)
        assert_allclose(np.squeeze(dw), wsoln)


class TestDSigmoid(unittest.TestCase):
    def test(self):
        T = data["sigmoid_backward"]
        z, soln = T
        dz = d_sigmoid(z)
        assert_allclose(np.squeeze(dz), soln)


class TestDCrossEntropy(unittest.TestCase):
    def test(self):
        T = data["ce_backward"]
        y, yh, soln = T
        db = d_cross_entropy_vec(y, yh)
        assert_allclose(np.squeeze(db), soln)


class TestForward(unittest.TestCase):
    def test_1(self):
        T1, T2 = data["forward_backward"]
        x, y, soln, _, _ = T1
        nn = NN(lr=1, n_epoch=1, weight_init_fn=zero_init, input_size=len(x),
                hidden_size=4, output_size=10)
        yh = forward(x, nn)
        assert_allclose(np.squeeze(yh), soln)

    def test_2(self):
        T1, T2 = data["forward_backward"]
        x, y, soln, _, _ = T2
        nn = NN(lr=1, n_epoch=1, weight_init_fn=random_init, input_size=len(x),
                hidden_size=4, output_size=10)
        yh = forward(x, nn)
        assert_allclose(np.squeeze(yh), soln)


class TestBackward(unittest.TestCase):
    def test_1(self):
        T1, T2 = data["forward_backward"]
        x, y, soln_yh, soln_d_w1, soln_d_w2 = T1
        nn = NN(lr=1, n_epoch=1, weight_init_fn=zero_init, input_size=len(x),
                hidden_size=4, output_size=10)
        yh = forward(x, nn)
        d_w1, d_w2 = backward(x, y, yh, nn)
        assert_allclose(np.squeeze(d_w1), soln_d_w1)
        assert_allclose(np.squeeze(d_w2), soln_d_w2)
    
    def test_2(self):
        T1, T2 = data["forward_backward"]
        x, y, soln_yh, soln_d_w1, soln_d_w2 = T2
        nn = NN(lr=1, n_epoch=1, weight_init_fn=random_init, input_size=len(x),
                hidden_size=4, output_size=10)
        yh = forward(x, nn)
        d_w1, d_w2 = backward(x, y, yh, nn)
        assert_allclose(np.squeeze(d_w1), soln_d_w1)
        assert_allclose(np.squeeze(d_w2), soln_d_w2)