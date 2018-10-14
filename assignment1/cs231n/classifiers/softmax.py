import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X, W)
  f = f - np.max(f, axis=1).reshape((N,1))
  L = 0
  for i in range(N):
    L += -np.log((np.exp(f[i, y[i]]) / (np.sum(np.exp(f[i]), axis=0, keepdims=True))))

  data_loss = (1.0 / N) * L 
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss

  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  dscores = probs
  # Simplified gradient function. Decrement the correct class prob by 1. Other probabilities unchanged
  dscores[np.arange(N), y] -= 1
  dscores /= N
  # Backprop over the multiplication in the loss function equation
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  f = np.dot(X, W)
  f = f - np.max(f, axis=1).reshape((N,1))
  win_probs = f[np.arange(N), y]
  win_probs = np.exp(win_probs)
  data_loss = np.sum(-np.log(win_probs / np.sum(np.exp(f), axis=1)))
  data_loss /= N
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss

  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  dscores = probs
  # Simplified gradient function. Decrement the correct class prob by 1. Other probabilities unchanged
  dscores[np.arange(N), y] -= 1
  dscores /= N
  # Backprop over the multiplication in the loss function equation
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

