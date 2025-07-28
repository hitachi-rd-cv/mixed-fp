import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch import nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


import torch

from lib_common.torch.autograd import mygrad, myjacobian, myjvp
from scipy.stats import gamma, beta

class MyMNISTBase(Dataset):
    DatasetClass = None

    def __init__(self, train=True, scale_noise=1., noise_type='gaussian', scaler='minmax'):
        self.mnist = self.DatasetClass(root='./data', train=train, download=True)

        # normalize data to [-1, 1] adding gaussian noise
        dtype = torch.get_default_dtype()
        data = self.mnist.data.to(dtype)
        if scaler == 'minmax':
            data = data / 255.0 * 2.0 - 1.0
        elif scaler == 'minmax2':
            data = data / 255.0
        elif scaler == 'standard':
            mean = data.mean(0, keepdim=True)
            std = data.std(0, unbiased=False, keepdim=True)
            data -= mean
            data /= std
        elif scaler == 'standard_flat':
            mean = data.mean(dim=None)
            std = data.std(dim=None, unbiased=False)
            data -= mean
            data /= std
        else:
            raise ValueError(f'Unknown scaler: {scaler}')

        if noise_type == 'gaussian':
            self.data = data + scale_noise * torch.randn_like(self.mnist.data.to(dtype))
        elif noise_type == 'uniform':
            self.data = data + scale_noise * (2.0 * torch.rand_like(self.mnist.data.to(dtype)) - 1.0)
        else:
            raise ValueError(f'Unknown noise_type: {noise_type}')
        self.targets = self.mnist.targets.long()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MyFashionMNIST(MyMNISTBase):
    DatasetClass = datasets.FashionMNIST












class BinaryAdultDataset(Dataset):
    def __init__(self, train=True, name_scaler='standard'):
        # Load the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

        # Handle missing values
        df.dropna(inplace=True)

        # Separate features and target
        X = df.drop("income", axis=1)
        y = df["income"]

        # Process features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        X[categorical_cols] = X[categorical_cols].apply(LabelEncoder().fit_transform)

        numerical_cols = X.columns.tolist()
        # Convert to tensors
        if name_scaler == 'standard':
            # Scale features using StandardScaler
            X[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])
        elif name_scaler == 'minmax':
            # Scale features using MinMaxScaler
            X[numerical_cols] = MinMaxScaler().fit_transform(X[numerical_cols])
        elif name_scaler == 'none':
            pass
        else:
            raise ValueError(f'Unknown scaler: {name_scaler}')

        self.data = torch.tensor(X.values, dtype=torch.get_default_dtype())
        self.targets = torch.tensor((y == ">50K").astype(int).values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]








class RegressionCaliforniaHousing(Dataset):
    def __init__(self, train=True, name_scaler='standard'):
        # Load the California housing dataset
        housing = datasets.fetch_california_housing()
        X_all = housing.data
        y_all = housing.target

        if name_scaler == 'standard':
            # Scale features using StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all)
        elif name_scaler == 'minmax':
            # Scale features using MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_all)
        elif name_scaler == 'none':
            X_scaled = X_all
        else:
            raise ValueError(f'Unknown scaler: {name_scaler}')

        # Convert features and targets to torch tensors
        self.data = torch.tensor(X_scaled, dtype=torch.get_default_dtype())
        self.targets = torch.tensor(y_all, dtype=torch.get_default_dtype())

        self.num_features = self.data.shape[1]  # Number of features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class IndexDataset(Dataset):
    def __init__(self, dataset_size):
        self.indices = list(range(dataset_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, scale_reg=None):
        super(LogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.scale_reg = scale_reg

    def forward(self, x):
        out = torch.sigmoid(torch.matmul(x, self.weight))
        return out.view(-1, 1)

    def loss(self, x, y, loss_weights=None, transform=None):
        if transform is not None:
            x = x @ transform

        out = self(x)
        if loss_weights is None:
            loss = nn.BCELoss()(out, y)
        else:
            assert len(loss_weights) == len(y)
            losses = nn.BCELoss(reduction='none')(out, y)
            loss = torch.mean(loss_weights * losses)

        if self.scale_reg is not None:
            l2_reg = self.scale_reg * torch.sum(self.weight ** 2) / 2.0
            loss += l2_reg

        return loss

    def metric(self, x, y):
        # Computes accuracy for binary classification
        with torch.no_grad():
            preds = (self(x) > 0.5).long().view(-1)
            return (preds == y.view(-1).long()).float().mean().item()

class MultiLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, scale_reg=None):
        super(MultiLogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale_reg = scale_reg
        self.weight = nn.Parameter(torch.randn(input_dim * output_dim) * 0.01)

    def forward(self, x):
        out = torch.sigmoid(torch.matmul(x, self.weight.reshape(self.input_dim, self.output_dim)))
        return out

    def loss(self, x, y, loss_weights=None, transform=None):
        if transform is not None:
            x = x @ transform

        out = self(x)
        if loss_weights is None:
            loss = nn.CrossEntropyLoss()(out, y.reshape(-1).long())
            # get one hot encoding
        else:
            losses = nn.CrossEntropyLoss(reduction='none')(out, y.reshape(-1).long())
            loss = torch.mean(loss_weights * losses)

        if self.scale_reg is not None:
            l2_reg = self.scale_reg * torch.sum(self.weight ** 2) / 2.0
            loss += l2_reg

        return loss

    def metric(self, x, y):
        # Computes accuracy for multiclass classification
        with torch.no_grad():
            preds = self(x).argmax(dim=1)
            return (preds == y.view(-1).long()).float().mean().item()


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1, scale_reg=None):
        """
        Args:
            input_dim (int): Number of features in the input.
            output_dim (int): Number of outputs. Default is 1.
            scale_reg (float or None): Scaling factor for L2 regularization. If None, no regularization is applied.
        """
        super(LinearRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(input_dim * output_dim) * 0.01)
        self.scale_reg = scale_reg

    def forward(self, x):
        """
        Forward pass through the linear layer.
        Args:
            x (Tensor): Input tensor of shape (N, input_dim).
        Returns:
            Tensor: Output tensor of shape (N, output_dim).
        """
        return torch.matmul(x, self.weight.reshape(self.input_dim, self.output_dim))

    def loss(self, x, y, loss_weights=None):
        """
        Computes the mean squared error (MSE) loss between the model output and target.
        Optionally adds L2 regularization on the weights.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Target tensor.
            loss_weights (Tensor, optional): Optional tensor of instance weights for the MSE loss.

        Returns:
            Tensor: Scalar loss value.
        """
        predictions = self(x)
        # Ensure target tensor matches the predictions' shape.
        y = y.view_as(predictions)

        if loss_weights is None:
            mse_loss = nn.MSELoss()(predictions, y)
        else:
            mse_loss = ((predictions - y) ** 2 * loss_weights.view_as(predictions)).mean()

        if self.scale_reg is not None:
            # L2 regularization on the weights
            reg_loss = self.scale_reg * torch.sum(self.weight ** 2) / 2.0
            mse_loss += reg_loss

        return mse_loss

    def metric(self, x, y):
        # Computes L2 error for regression
        with torch.no_grad():
            preds = self(x)
            return torch.sqrt(torch.mean((preds - y.view_as(preds)) ** 2)).item()

class HyperSingleRegularization(nn.Module):
    def __init__(self, model, initial_reg_params=None, scale_reg=0.1):
        super().__init__()
        self.model = model
        self.scale_reg = scale_reg

        if initial_reg_params is None:
            initial_reg_params = torch.ones(1)
        self.reg_params = nn.Parameter(initial_reg_params)

        self.inner_param = self.model.weight
        self.outer_param = self.reg_params

    def regularize(self, weights):
        l2_reg = self.scale_reg * (self.reg_params ** 2) * torch.sum(weights ** 2) / 2.0
        return l2_reg

    def inner_loss(self, data, target):
        loss = self.model.loss(data, target)
        reg_val = self.regularize(self.model.weight)
        total_loss = loss + reg_val
        return total_loss

    def outer_loss(self, data, target):
        return self.model.loss(data, target)

    def metric(self, data, target):
        return self.model.metric(data, target)

class HyperMultiRegularization(nn.Module):
    def __init__(self, model, initial_reg_params=None, scale_reg=0.1):
        super().__init__()
        self.model = model
        self.scale_reg = scale_reg

        if initial_reg_params is None:
            initial_reg_params = torch.ones_like(self.model.weight)
        self.reg_params = nn.Parameter(initial_reg_params)

        self.inner_param = self.model.weight
        self.outer_param = self.reg_params

    def inner_loss(self, data, target):
        loss = self.model.loss(data, target)
        reg_val = self.regularize(self.model.weight)
        total_loss = loss + reg_val

        return total_loss

    def outer_loss(self, data, target):
        return self.model.loss(data, target)

    def metric(self, data, target):
        return self.model.metric(data, target)

    def regularize(self, weights):
        l2_reg = self.scale_reg * torch.sum((self.reg_params ** 2) * weights ** 2) / 2.0
        return l2_reg

class HyperInstanceLossWeight(nn.Module):
    def __init__(self, model, n_train, activation='none', scale_outer_reg=0.0):
        super().__init__()
        self.model = model
        self.activation = activation
        self.scale_reg = scale_outer_reg
        self.inner_param = self.model.weight

        if activation == 'none':
            self.outer_param = nn.Parameter(torch.ones(n_train))
        elif activation == 'sigmoid':
            self.outer_param = nn.Parameter(torch.zeros(n_train))
        elif activation == 'tanh':
            self.outer_param = nn.Parameter(torch.zeros(n_train))
        elif activation == 'softplus':
            self.outer_param = nn.Parameter(torch.zeros(n_train))
        else:
            raise ValueError(f'Unknown activation: {activation}')


    def inner_loss(self, data, target, indices):
        if self.activation == 'none':
            loss_weights = self.outer_param[indices]
        elif self.activation == 'sigmoid':
            loss_weights = torch.sigmoid(self.outer_param[indices]) * 2
        elif self.activation == 'tanh':
            loss_weights = torch.tanh(self.outer_param[indices]) + 1.
        elif self.activation == 'softplus':
            loss_weights = torch.nn.functional.softplus(self.outer_param[indices])
        else:
            raise ValueError(f'Unknown activation: {self.activation}')
        print(f'loss_weights: {loss_weights}')
        return self.model.loss(data, target, loss_weights=loss_weights)

    def outer_loss(self, data, target):
        if self.scale_reg > 0:
            reg_loss = self.scale_reg * torch.sum(self.outer_param ** 2) / 2.0
            return self.model.loss(data, target) + reg_loss
        else:
            return self.model.loss(data, target)

    def metric(self, data, target):
        return self.model.metric(data, target)

class HyperRepresentation(nn.Module):
    def __init__(self, model, rep_dim, input_dim):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.rep_dim = rep_dim

        self.weight = nn.Parameter(torch.randn(rep_dim * input_dim) * 0.01)

        self.inner_param = self.model.weight
        self.outer_param = self.weight

    def inner_loss(self, data, target):
        # reshape self.weight to input_dim x rep_dim
        transform = self.weight.reshape(self.input_dim, self.rep_dim)
        return self.model.loss(data, target, transform=transform)

    def outer_loss(self, data, target):
        transform = self.weight.reshape(self.input_dim, self.rep_dim)
        return self.model.loss(data, target, transform=transform)

    def metric(self, data, target):
        return self.model.metric(data, target)

class HyperMetaLearning(nn.Module):
    def __init__(self, model, beta=1, initializer='weight', scale_reg=0.0):
        super().__init__()
        self.model = model
        self.beta = beta
        self.initializer = initializer
        self.scale_reg = scale_reg
        self.inner_param = self.model.weight
        if initializer == 'weight':
            self.outer_param = nn.Parameter(self.model.weight.clone().detach())
        elif initializer == 'random':
            self.outer_param = nn.Parameter(torch.randn_like(self.model.weight) * 0.01)
        elif initializer == 'zeros':
            self.outer_param = nn.Parameter(torch.zeros_like(self.model.weight))
        else:
            raise ValueError(f'Unknown initializer: {initializer}')

    def inner_loss(self, data, target):
        return self.model.loss(data, target) + self.beta * torch.sum((self.outer_param - self.inner_param) ** 2) / 2.0

    def outer_loss(self, data, target):
        if self.scale_reg > 0:
            reg_loss = self.scale_reg * torch.sum(self.outer_param ** 2) / 2.0
        else:
            reg_loss = 0.0

        return self.model.loss(data, target) + reg_loss
    
    def metric(self, data, target):
        return self.model.metric(data, target)

class DummyOracle:
    def __init__(self, n, As, B, c, d):
        self.n = n

        self.A_parent = As
        self.A_true = torch.mean(As, dim=0)
        self.B_true = B
        self.c_true = c
        self.d_true = d
        self.A_inv_true = torch.linalg.inv(self.A_true)
        self.v_true = - self.B_true @ self.A_inv_true @ self.c_true + self.d_true

        self.indices_train = np.random.choice(len(As), n, replace=True)

    @classmethod
    def sample_dummies(cls, size_x, size_lambda, n_parent, epsilon, dist='uniform', mu=0.5):
        A_parent = torch.zeros((n_parent, size_x, size_x))
        for i in range(n_parent):
            matrix = torch.randn(size_x, size_x)
            symmetric_matrix = (matrix + matrix.T) / 2
            eigenvectors, _ = torch.linalg.qr(symmetric_matrix)
            if dist == 'uniform':
                eigenvalues = torch.rand(size_x) * (1 - epsilon)
            elif dist == 'gamma':
                # μ: mean, epsilon: shape
                scale = mu / epsilon
                eigenvalues = gamma.rvs(a=epsilon, scale=scale, size=size_x)
                eigenvalues = torch.tensor(eigenvalues, dtype=eigenvectors.dtype)
            elif dist == 'beta':
                a, b = cls.compute_beta_params(mu, epsilon)
                eigenvalues = beta.rvs(a=a, b=b, size=size_x)
                eigenvalues = torch.tensor(eigenvalues, dtype=eigenvectors.dtype)
            else:
                raise ValueError(f'Unknown dist: {dist}')

            A = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            A_parent[i] = A

        B = torch.rand(size_lambda, size_x)
        c = torch.randn(size_x)
        d = torch.randn(size_lambda)
        return A_parent, B, c, d

    @staticmethod
    def compute_beta_params(mean, var):
        # Ensure that the variance is valid: var must be <= mean*(1-mean)
        if var >= mean * (1 - mean):
            raise ValueError("Invalid variance for the given mean. Variance must be less than mean*(1-mean).")
        # Calculate the sum of the parameters
        temp = mean * (1 - mean) / var - 1
        alpha = mean * temp
        beta_param = (1 - mean) * temp
        return alpha, beta_param

    def sample_Av(self, step, v, explicit=False):
        idx = self.indices_train[step]
        A = self.A_parent[idx]
        return A @ v

    def get_v_error(self, v):
        return torch.norm(v - self.v_true)


class GradientOracle:
    def __init__(self, bilevel_model, dataset_train, dataset_val, batch_size, depth,
            input_dim=784, device='cuda', shuffle=True, linalg_inverse=True,
            depth_true=100000, scale_true=0.1, precompute_As=False, compute_v_true=True):
        self.device = device
        self.bilevel_model = bilevel_model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.input_dim = input_dim
        self.n_train = len(dataset_train)
        self.n_val = len(dataset_val)

        self.all_indices_train = np.arange(self.n_train)
        self.all_indices_val = np.arange(self.n_val)

        if shuffle:
            self.minibatch_indices_train = np.random.choice(self.n_train, (depth, batch_size), replace=True)
        else:
            assert self.n_train == batch_size
            self.minibatch_indices_train = np.array([np.arange(self.n_train)] * depth)

        dtype = torch.get_default_dtype()

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.n_train, shuffle=False)
        data_train, target_train = next(iter(train_loader))
        self.data_train = data_train.to(self.device).view(-1, input_dim).to(dtype)
        self.target_train = target_train.to(self.device).view(-1, 1).to(dtype)

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.n_val, shuffle=False)
        data_val, target_val = next(iter(val_loader))
        self.data_val = data_val.to(self.device).view(-1, input_dim).to(dtype)
        self.target_val = target_val.to(self.device).view(-1, 1).to(dtype)

        if precompute_As:
            print('Precomputing As')
            self.As = []
            for t in tqdm(range(depth)):
                A = self.sample_A(t)
                self.As.append(A)

        self.B_true = self.sample_inner_jacobian(self.bilevel_model.outer_param, self.all_indices_train)
        self.c_true = self.sample_outer_grad(self.bilevel_model.inner_param, self.all_indices_val)
        self.d_true = self.sample_outer_grad(self.bilevel_model.outer_param, self.all_indices_val)

        if compute_v_true:
            A_true = self.sample_inner_jacobian(self.bilevel_model.inner_param, self.all_indices_train)

            if linalg_inverse:
                A_inv_true = torch.tensor(np.linalg.inv(A_true.cpu().numpy()), device=self.device, dtype=dtype)

            else:
                I = torch.eye(input_dim, device=self.device, dtype=dtype)
                Z_n = I.clone().detach()
                A_inv_true = I.clone().detach()
                print('Computing A_inv_true')
                for _ in tqdm(range(depth_true)):
                    Z_n = Z_n @ (I - scale_true * A_true)
                    A_inv_true += Z_n * scale_true

            self.v_true = - self.B_true @ A_inv_true @ self.c_true + self.d_true
        else:
            self.v_true = None

    def sample_inner_jacobian(self, input, indices):
        data = self.data_train[indices]
        target = self.target_train[indices]

        self.bilevel_model.zero_grad()
        if isinstance(self.bilevel_model, HyperInstanceLossWeight):
            loss = self.bilevel_model.inner_loss(data, target, indices)
        else:
            loss = self.bilevel_model.inner_loss(data, target)

        grads = mygrad(loss, self.bilevel_model.inner_param, create_graph=True)
        jacobian = myjacobian(grads, input)
        return jacobian[0]

    # @profile
    def sample_inner_jvp(self, v, input, indices):
        data = self.data_train[indices]
        target = self.target_train[indices]

        self.bilevel_model.zero_grad()
        if isinstance(self.bilevel_model, HyperInstanceLossWeight):
            loss = self.bilevel_model.inner_loss(data, target, indices)
        else:
            loss = self.bilevel_model.inner_loss(data, target)

        grads = mygrad(loss, self.bilevel_model.inner_param, create_graph=True)
        jvp = myjvp(grads, input, v)
        return jvp

    # @profile
    def sample_inner_mvp(self, v, input, indices, scale):
        data = self.data_train[indices]
        target = self.target_train[indices]

        self.bilevel_model.zero_grad()
        if isinstance(self.bilevel_model, HyperInstanceLossWeight):
            loss = self.bilevel_model.inner_loss(data, target, indices)
        else:
            loss = self.bilevel_model.inner_loss(data, target)

        grads = mygrad(loss, self.bilevel_model.inner_param, create_graph=True)
        mapping = input - scale * grads
        mvp = myjvp(mapping, input, v)
        return mvp

    def sample_outer_grad(self, input, indices):
        data = self.data_val[indices]
        target = self.target_val[indices]

        self.bilevel_model.zero_grad()
        loss = self.bilevel_model.outer_loss(data, target)

        grads = mygrad(loss, input, allow_unused=True)
        if grads[0] is None:
            grads = list(grads)
            grads[0] = torch.zeros_like(input)
            grads = tuple(grads)
        return grads[0]


    def sample_A(self, step):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]
        A = self.sample_inner_jacobian(self.bilevel_model.inner_param, indices)
        return A

    def sample_A_sym(self, step):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]
        A = self.sample_inner_jacobian(self.bilevel_model.inner_param, indices)
        return (A + A.T) * 0.5

    def sample_Av(self, step, v, explicit=False):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]

        if explicit:
            A = self.sample_inner_jacobian(self.bilevel_model.inner_param, indices)
            A = (A + A.T) * 0.5
            return A @ v
        else:
            return self.sample_inner_jvp(v, self.bilevel_model.inner_param, indices)

    def sample_Zv(self, step, v, scale, explicit=False):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]

        if explicit:
            A = self.sample_inner_jacobian(self.bilevel_model.inner_param, indices)
            A = (A + A.T) * 0.5
            I = torch.eye(self.input_dim, device=self.device, dtype=A.dtype)
            return  (I - scale * A) @ v
        else:
            return self.sample_inner_mvp(v, self.bilevel_model.inner_param, indices, scale)

    def sample_Bv(self, step, v):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]
        return self.sample_inner_jvp(v, self.bilevel_model.outer_param, indices)

    def sample_c(self, step):
        if step < 0:
            indices = self.all_indices_val
        else:
            indices = self.minibatch_indices_train[step]
        return self.sample_outer_grad(self.bilevel_model.inner_param, indices)

    def sample_d(self, step):
        if step < 0:
            indices = self.all_indices_train
        else:
            indices = self.minibatch_indices_train[step]
        return self.sample_outer_grad(self.bilevel_model.outer_param, indices)

    def get_v_error(self, v):
        if self.v_true is None:
            return -1.
        else:
            return torch.linalg.norm(v - self.v_true)


def kahan_sum(sum_so_far, value_to_add, correction):
    y = value_to_add - correction
    t = sum_so_far + y
    correction = (t - sum_so_far) - y
    return t, correction


def normalize(vector):
    norm = torch.linalg.norm(vector)
    return vector / (norm + 1e-8)  # Add a small constant to avoid division by zero


def rescale(u, v):
    u_norm = torch.linalg.norm(u)
    if u_norm > 1e5 or u_norm < 1e-5:
        scale_factor = 1.0 / u_norm
        return u * scale_factor, v * scale_factor
    return u, v


def get_learning_rate(t: int, params: dict, scheduler: str = 'linear') -> float:
    """Calculate learning rate based on scheduler type and parameters.

    Args:
        t (int): Current iteration (0-indexed)
        params (dict): Dictionary containing learning rate parameters
            - 'lr': Fixed learning rate (used if directly specified)
            - 'alpha', 'gamma': Base parameters for scheduling learning rates
            - 'p': Polynomial degree for Polynomial Decay (optional)
            - 'end_lr': Final learning rate for Polynomial Decay (optional)
        scheduler (str): Type of scheduler
            - 'const': Constant learning rate (returns params['alpha'])
            - 'linear': Linear decay (1 / (t+1) type)
            - 'root': 1 / sqrt(t+1) type decay
            - 'exp': Exponential Decay
            - 'poly': Polynomial Decay

    Returns:
        float: Learning rate for the current iteration
    """
    # If 'lr' is specified, use it as the fixed learning rate
    if "lr" in params:
        # When using 'lr', ensure 'alpha' and 'gamma' are not specified
        assert 'alpha' not in params and 'gamma' not in params, \
            "When 'lr' is specified, do not provide 'alpha' or 'gamma'."
        return params['lr']

    # If 'lr' is not specified, ensure 'alpha' and 'gamma' are provided
    assert 'alpha' in params and 'gamma' in params, \
        "lr or (alpha and gamma) must be specified in params."

    alpha = params['alpha']
    gamma = params['gamma']

    if scheduler == 'const':
        # Constant learning rate
        return alpha

    elif scheduler == 'linear':
        # Linear decay: lr = alpha / (gamma + t + 1)
        return alpha / (gamma + t + 1)

    elif scheduler == 'root':
        # 1 / sqrt(t+1) decay: lr = alpha / sqrt(gamma + t + 1)
        return alpha / math.sqrt(gamma + t + 1)

    elif scheduler == 'exp':
        # Exponential decay
        # Typically: lr = alpha * exp(- decay_rate * t)
        # Here, gamma is used as decay_rate
        return alpha * math.exp(- gamma * t)

    elif scheduler == 'poly':
        # Polynomial decay
        # Example: lr = (alpha - end_lr) * (1 - t / T)^p + end_lr
        # Simplified: lr = alpha * (1 - t/T)^p
        # T (total steps), p, and end_lr are typically provided as parameters:

        p = params.get('p', 2.0)  # Degree of the polynomial (default is 2)
        end_lr = params.get('end_lr', 0.0)  # Final learning rate (default is 0)

        # Here, gamma is used as the "total steps (T)"
        # Ensure (1 - t/T) is not negative by capping progress at 1.0
        progress = min(t / gamma, 1.0)

        return (alpha - end_lr) * ((1.0 - progress) ** p) + end_lr

    else:
        raise ValueError(f'Unknown scheduler: {scheduler}')


def estimate_hypergradient(oracle, depth, method, params=None, _run=None, silent=False, compute_v_true=True):
    if params is None:
        params = {}

    B = oracle.B_true
    c = oracle.c_true
    d = oracle.d_true
    v = d
    u = c
    w = torch.zeros_like(c)
    w_ = c
    z = torch.zeros_like(c)

    v_errors = []
    
    scale = params['scale']

    if silent:
        pbar = range(depth)
    else:
        pbar = tqdm(range(depth))
    for t in pbar:
        if method == 'neumann':
            Aw = oracle.sample_Av(t, w)
            v = - scale * B @ w + d
            w = w - scale * Aw + c
        elif method == 'unroll':
            Au = oracle.sample_Av(t, u)
            v = v - scale * B @ u
            u = u - scale * Au
        elif method == 'grazzi':
            Aw = oracle.sample_Av(t, w)
            lr = get_learning_rate(t, params, params['scheduler'])
            v = - scale * B @ w + d
            w = w + lr * (- scale * Aw + c)
        elif method == 'grazzi_scaled':
            Aw = oracle.sample_Av(t, w)
            lr = get_learning_rate(t, params, params['scheduler'])
            v = - scale * B @ w + d
            w = w + lr * (- scale * Aw + c)
        elif method == 'vr':
            Aw = oracle.sample_Av(t, w_)
            Au = oracle.sample_Av(t, u)
            a = params['a']
            v = - scale * B @ z + d
            z = (1 - a) * (u + z) + a * w_
            u = u - scale * Au
            w_ = w_ - scale * Aw + c
        elif method == 'vr_km':
            Aw = oracle.sample_Av(t, w_)
            Au = oracle.sample_Av(t, u)
            a = params['a']
            lr = get_learning_rate(t, params, params['scheduler'])
            v = - scale * B @ z + d
            # (1-lr)*z+lr*((1-a)*(u+z)+a*w)
            z = z + lr * ((1 - a) * (u + z) + a * w_ - z)
            # (1. - lr) * w + lr * ((I - A) @ w + c) = w + lr * (- A @ w + c)
            w_ = w_ + lr * (- scale * Aw + c)
            # (1 - lr) * u + lr * (I - A) @ u = u - lr * A @ u
            u = u + lr * (- scale * Au)
        else:
            raise ValueError(f'Unknown method: {method}')

        if compute_v_true:
            v_error = oracle.get_v_error(v).item()

            # Check if v_error is NaN and raise ValueError if it is
            if torch.isnan(torch.tensor(v_error)):
                raise ValueError(f"NaN encountered in v_error at iteration {t}")

            if _run:
                _run.log_scalar('v_error', v_error)
            v_errors.append(v_error)

        if not silent:
            if compute_v_true:
                pbar.set_description(f'v_error: {v_error:.4f}')
            else:
                pbar.set_description('v_error: Not computed')

    return v, v_errors

