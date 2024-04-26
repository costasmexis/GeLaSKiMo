import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def sample_hidden_given_visible(
    weight_ml : np.array,
    mu_ml : np.array,
    var_ml : float,
    visible_samples : np.array
    ) -> np.array:

    q = weight_ml.shape[1]
    m = np.transpose(weight_ml) @ weight_ml + var_ml * np.eye(q)

    cov = var_ml * np.linalg.inv(m) @ np.eye(q)
    act_hidden = []
    for data_visible in visible_samples:
        mean = np.linalg.inv(m) @ np.transpose(weight_ml) @ (data_visible - mu_ml)
        sample = np.random.multivariate_normal(mean.real, cov.real, size=1)
        act_hidden.append(sample[0])

    return np.array(act_hidden)

def sample_visible_given_hidden(
    weight_ml : np.array,
    mu_ml : np.array,
    var_ml : float,
    hidden_samples : np.array
    ) -> np.array:

    d = weight_ml.shape[0]

    act_visible = []
    for data_hidden in hidden_samples:
        mean = weight_ml @ data_hidden + mu_ml
        cov = var_ml * np.eye(d)
        sample = np.random.multivariate_normal(mean.real,cov.real,size=1)
        act_visible.append(sample[0])

    return np.array(act_visible)

def generative_ppca(df: pd.DataFrame, q: int = None) -> pd.DataFrame:
    dfcolumns = df.columns
    data = df.values
    d = data.shape[1]

    mu_ml = np.mean(data, axis=0)
    data_cov = np.cov(data, rowvar=False)

    if q is None:
        q = int(df.shape[1] / 2)

    # Variance
    lambdas, eigenvecs = np.linalg.eig(data_cov)
    idx = lambdas.argsort()[::-1]
    lambdas = lambdas[idx]
    eigenvecs = - eigenvecs[:,idx]

    var_ml = (1.0 / (d-q)) * sum([lambdas[j] for j in range(q,d)])

    # Weight matrix
    uq = eigenvecs[:,:q]
    lambdaq = np.diag(lambdas[:q])
    weight_ml = uq @ np.sqrt(lambdaq - var_ml * np.eye(q))
    
    act_hidden = sample_hidden_given_visible(
        weight_ml=weight_ml,
        mu_ml=mu_ml,
        var_ml=var_ml,
        visible_samples=data
        )

    mean_hidden = np.full(q,0)
    cov_hidden = np.eye(q)

    no_samples = len(data)
    samples_hidden = np.random.multivariate_normal(mean_hidden,cov_hidden,size=no_samples)

    generated_samples = sample_visible_given_hidden(
        weight_ml=weight_ml,
        mu_ml=mu_ml,
        var_ml=var_ml,
        hidden_samples=samples_hidden
        )
    
    generated_samples_df = pd.DataFrame(generated_samples, columns=dfcolumns)

    return generated_samples_df