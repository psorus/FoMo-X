import torch
import torch.distributions as distributions
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score



def log_uniform(lower, upper):
    return torch.exp(torch.rand(1) * (torch.log(torch.tensor(upper)) - torch.log(torch.tensor(lower))) + torch.log(torch.tensor(lower))).item()

def lin_uniform(lower, upper):
    return torch.rand(1) * (upper - lower) + lower

def log_tuple(tpl):
    if isinstance(tpl, tuple) or isinstance(tpl, list):
        return log_uniform(tpl[0], tpl[1])#.numpy()
    else:
        return tpl
def lin_tuple(tpl):
    if isinstance(tpl, tuple) or isinstance(tpl, list):
        return lin_uniform(tpl[0], tpl[1])#.numpy()
    else:
        return tpl

def random_covariance(feature_dim, variance_scale, variance_uniformity):
    L_matrix = torch.randn(feature_dim, feature_dim) * variance_scale
    L = torch.tril(L_matrix)
    L.diagonal().abs_().add_(1e-5+variance_uniformity*variance_scale)
    cov = L @ L.T
    cov = 0.5* (cov + cov.T)  # Ensure symmetry
    eigenv=torch.linalg.eigvalsh(cov)
    toadd=eigenv.min()
    count=100
    while count>0 and (toadd:=torch.linalg.eigvalsh(cov).min())<1e-5:
        cov = cov - 10*(toadd-1e-5)*torch.eye(feature_dim)
        count-=1
    if count<=0:
        return random_covariance(feature_dim, variance_scale, variance_uniformity)
    jitter = 1e-2 * torch.eye(feature_dim)
    cov += jitter  # Add jitter for numerical stability
    return cov

def generate_random_gmm(max_components, feature_dim, device=None, max_mean=5.0, min_variance_scale=0.25,max_variance_scale=2.0, min_variance_uniformity=0.1,max_variance_uniformity=1.0):
    max_mean=lin_tuple(max_mean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    variance_scale=log_uniform(min_variance_scale, max_variance_scale)
    variance_uniformity=log_uniform(min_variance_uniformity, max_variance_uniformity)
    num_components = torch.randint(1, max_components + 1, (1,)).item()
    alpha = torch.ones(num_components) * 1.0 # Symmetric prior
    mixing_probs = distributions.Dirichlet(alpha).sample()
    min_val, max_val = -max_mean, max_mean
    means = (max_val - min_val) * torch.rand(num_components, feature_dim) + min_val
    covariances = []
    for _ in range(num_components):
        cov=random_covariance(feature_dim, variance_scale, variance_uniformity)
        covariances.append(cov.unsqueeze(0))
    covariances = torch.cat(covariances, dim=0)

    feature_sub_dim=feature_dim if torch.randint(0,2,(1,)).item()==0 else torch.randint(1, min(20, feature_dim+1), (1,)).item()
    subdims=sorted(np.random.choice(feature_dim, feature_sub_dim, replace=False))
    return {"num_components": num_components,
            "mixing_probs": mixing_probs.to(device),
            "means": means.to(device),
            "covariances": covariances.to(device),
            "subdims": subdims}

def inflate(num_components, mixing_probs, means, covariances,subdims, factor=2):
    covariances2=covariances*factor
    return {"num_components": num_components,
            "mixing_probs": mixing_probs,
            "means": means,
            "covariances": covariances2,
            "subdims": subdims}


def generate_gmm_samples(num_samples, num_components, mixing_probs, means, covariances, subdims, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    feature_dim=len(means[0])

    categorical = distributions.Categorical(mixing_probs)
    indices = categorical.sample((num_samples,))
    samples = torch.zeros(num_samples, feature_dim, device=device)
    for k in range(num_components):
        component_mask = (indices == k)
        num_k_samples = component_mask.sum()

        if num_k_samples > 0:
            mvn = distributions.MultivariateNormal(
                loc=means[k], 
                covariance_matrix=covariances[k],
            )
            
            samples[component_mask] = mvn.sample((num_k_samples,))

    return samples, indices



def check_normal(samples, num_components, mixing_probs, means, covariances, subdims,border=None, extend=0.05, flip_extend=False, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    feature_dim=len(means[0])
    cov_use=covariances[:,subdims,:][:,:,subdims]
    means_use=means[:,subdims]
    samples_use=samples[:,subdims]
    if len(subdims)>1 or True:
        mvns = [distributions.MultivariateNormal(loc=means_use[k], covariance_matrix=cov_use[k]) for k in range(num_components)]
    else:
        mvns = [distributions.Normal(loc=means[k,subdims], scale=torch.sqrt(covariances[k, subdims, subdims])) for k in range(num_components)]
    probs = torch.zeros(samples.shape[0], num_components, device=device)
    for k in range(num_components):
        probs[:, k] = mvns[k].log_prob(samples_use)
    if feature_dim<25 and False:#exact calculation inefficient in high dimensions because of numerical instability
        probs=torch.sum(torch.exp(probs)*mixing_probs, dim=1)
    else:
        probs=torch.max(probs+torch.log(mixing_probs), dim=1)[0]
    if border is None:
        border=probs.kthvalue(int(0.01*len(probs)))[0]
    if flip_extend:
        extend=-extend
    index=probs>=border*(1+extend)
    return index, border
def min_distance(samples, num_components, mixing_probs, means, covariances, subdims,border=None, extend=0.05, flip_extend=False, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    feature_dim=len(means[0])
    cov_use=covariances[:,subdims,:][:,:,subdims]
    means_use=means[:,subdims]
    samples_use=samples[:,subdims]
    if len(subdims)>1 or True:
        mvns = [distributions.MultivariateNormal(loc=means_use[k], covariance_matrix=cov_use[k]) for k in range(num_components)]
    else:
        mvns = [distributions.Normal(loc=means[k,subdims], scale=torch.sqrt(covariances[k, subdims, subdims])) for k in range(num_components)]
    dists = torch.zeros(samples.shape[0], num_components, device=device)
    for k in range(num_components):
        dists[:, k] = mvns[k].log_prob(samples_use)
    dists, dex=torch.max(dists, dim=1)
    return dists, dex

def assert_length(func):
    def wrapper(N, *args, border=None, increase=1.5, **kwargs):
        data, border=func(int(N*increase), *args,**kwargs, border=border)
        tries=100
        while len(data)<N:
            tries-=1
            if tries<=0:
                raise RuntimeError("Too many tries to generate enough samples")
            togen=int(N*increase)-len(data)
            more_data, _ =func(togen, *args,**kwargs, border=border)#, show=increase>=10.0)
            increase+=0.5
            if len(more_data)==0:
                continue
            data=torch.cat([data, more_data], dim=0)
        return data[:N], border
    return wrapper

@assert_length
def generate_normal(N, model, delta,device=None, border=None, show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    if show or False: print("CALLED NORMAL",N)
    X, Z = generate_gmm_samples(N,**model, device=device)
    normal_index, border= check_normal(X,**model, extend=delta, flip_extend=False, border=border, device=device)
    X, Z= X[normal_index], Z[normal_index]
    return X, border
@assert_length
def generate_abnormal(N, model, delta, anomaly_inflation,device=None, border=None, show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    if show or False: print("CALLED ABNORMAL",N)
    X, Z = generate_gmm_samples(N,**inflate(**model, factor=anomaly_inflation), device=device)
    abnormal_index=~check_normal(X,border=border, extend=delta, flip_extend=True, **model, device=device)[0] 
    X, Z= X[abnormal_index], Z[abnormal_index]
    return X, border





max_components = 10   # Number of components
D = 100   # Feature dimension (for easy plotting)
N = 5000 # Number of samples
delta=-0.1
max_mean=5.0
min_variance_scale=0.25
max_variance_scale=2.0
min_variance_uniformity=0.1
max_variance_uniformity=1.0
anomaly_inflation=5.0
samples_per_batch=8
batch_count=10

def zero_pad(X, target_dim, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    current_dim=X.shape[1]
    if current_dim>=target_dim:
        return X
    padding=torch.zeros(X.shape[0], target_dim-current_dim, device=device)
    return torch.cat([X, padding], dim=1)

def severity(model, Xa, X):
    da, _=min_distance(Xa, **model)
    d, _=min_distance(X, **model)
    da, d=da.cpu().numpy(), d.cpu().numpy()
    border1=np.median(da)#smaller
    border2=np.median(d)#larger
    def apply_severity_normal(d, border=border2):
        severities=[]
        for dist in d:
            if dist <= border:
                severities.append([0,1,0,0])
            else:
                severities.append([1,0,0,0])
        return np.array(severities)
    def apply_severity_abnormal(d, border=border1):
        severities=[]
        for dist in d:
            if dist <= border1:
                severities.append([0,0,0,1])
            else:
                severities.append([0,0,1,0])
        return np.array(severities)
    return apply_severity_abnormal(da), apply_severity_normal(d)



def generate_dataset(samples,feature_dim,device=None,max_feature_dim=None,max_components=max_components, max_mean=max_mean, delta=delta, min_variance_scale=min_variance_scale, max_variance_scale=max_variance_scale, min_variance_uniformity=min_variance_uniformity, max_variance_uniformity=max_variance_uniformity, anomaly_inflation=anomaly_inflation):
    if max_feature_dim is None:
        max_feature_dim=max(feature_dim) if isinstance(feature_dim, (list, tuple)) else feature_dim
    feature_dim=int(log_tuple(feature_dim))
    if feature_dim>max_feature_dim:
        feature_dim=max_feature_dim

    delta=lin_tuple(delta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    if not isinstance(delta, torch.Tensor):
        delta=torch.tensor(delta)
    delta=delta.to(device)
    mixture=generate_random_gmm(max_components, feature_dim, device=device, max_mean=max_mean, min_variance_scale=min_variance_scale, max_variance_scale=max_variance_scale, min_variance_uniformity=min_variance_uniformity, max_variance_uniformity=max_variance_uniformity)
    X, border=generate_normal(samples, mixture, delta, device=device)
    Xa, _=generate_abnormal(samples, mixture, delta, anomaly_inflation, border=border, device=device)
    sevXa, sevX=severity(mixture, Xa, X)
    X, Xa=zero_pad(X, max_feature_dim, device=device), zero_pad(Xa, max_feature_dim, device=device)
    sevX=torch.tensor(sevX, device=device)
    sevXa=torch.tensor(sevXa, device=device)
    return X, Xa, sevX, sevXa

def generate_batch(count, samples, feature_dim, device=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
    batch_X=[]
    batch_Xa=[]
    batch_SevX=[]
    batch_SevXa=[]
    for _ in tqdm(range(count), desc="Generating batch"):
        while True:
            try:
                X, Xa, sevX, sevXa=generate_dataset(samples, feature_dim, device=device, **kwargs)
            except Exception as e:
                continue
            else:
                break
        batch_X.append(X)
        batch_Xa.append(Xa)
        batch_SevX.append(sevX)
        batch_SevXa.append(sevXa)

    return torch.stack(batch_X), torch.stack(batch_Xa), torch.stack(batch_SevX), torch.stack(batch_SevXa)
    
if True and __name__=="__main__":
    device="cpu"

    from time import time
    t0=time()
    
    X, Xa, sevX, sevXa =generate_dataset(N,2, device=device, delta=0.05)

    
    t1=time()
    print("Required time", t1-t0)
    
    print(f"Generated samples (X) shape: {X.shape}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    # Scatter plot, color-coded by the component index (Z)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c="green", s=20)
    plt.scatter(Xa[:, 0].numpy(), Xa[:, 1].numpy(), c="red", s=20)
    plt.title(f'Samples from a Random-Component GMM (D={D})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c="green", s=20, label="Normal")
    sevt=np.argmax(sevXa, axis=1)
    plt.scatter(Xa[:, 0].numpy(), Xa[:, 1].numpy(), c=sevt, s=20, label="Abnormal Severity")
    plt.show()


    seva=np.concatenate((np.argmax(sevX,axis=1), sevt), axis=0)

    print(np.histogram(seva, bins=4))






