import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets

from collections import namedtuple, OrderedDict
import numpy as np
import logging
import math
import os
import json
import requests

import imageio
from skimage import img_as_ubyte

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1E-20

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu)/std

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 


#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})


#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }


#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, img_as_ubyte(image))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def configure_logger(model_name, debug):
    logging.basicConfig(format='%(message)s')  # , level=logging.DEBUG)
    logger = logging.getLogger()
    logger.handlers = []  # remove the default logger

    # add a new logger for stdout
    formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    return logger

def to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def to_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def model_eval(model):
    model.eval()


def model_train(model):
    model.train()


def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def get_gaussian_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta = eps * torch.randn(*delta.shape)
    delta.requires_grad = requires_grad
    return delta


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def attack_pgd_training(model, X, y, eps, alpha, opt, attack_iters, rs=True, early_stopping=False):
    delta = torch.zeros_like(X).cuda()
    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(clamp(X + delta, 0, 1))
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        if early_stopping:
            idx_update = output.max(1)[1] == y
        else:
            idx_update = torch.ones(y.shape, dtype=torch.bool)
        grad_sign = sign(grad)

        delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
        delta.data = clamp(X + delta.data, 0, 1) - X
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()

    return delta.detach()


def attack_pgd(model, X, y, eps, alpha, opt, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            idx_update = output.max(1)[1] == y      # not update correctly classified exps for evaluation
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data[idx_update] = (delta + alpha * sign(grad))[idx_update]
            else:
                delta.data[idx_update] = (delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5)[idx_update]

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta


def rob_acc(batches, model, eps, pgd_alpha, opt, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, verbose=False, cuda=True):
    err_inds = []   # record the misclassified indices
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, attack_iters, n_restarts, rs=rs,
                               verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        if corner:
            pgd_delta = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X
        pgd_delta_proj = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
    
    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    return robust_acc, avg_loss


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        n = module.in_features
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
    elif lr_schedule_type == 'flat':
        lr_schedule = lambda t: lr_max
    elif lr_schedule_type == 'piecewise':
        def lr_schedule(t):
            if t / n_epochs < 0.5:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10
            else:
                return lr_max / 100
    else:
        raise ValueError('wrong lr_schedule_type')
    
    return lr_schedule


def backward(loss, opt):
    loss.backward()


#### for awp adversary
def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


#### for robustbench models
def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_model(model_name, model_dir='./models', norm='Linf'):
    from model_zoo_old.models import model_dicts as all_models
    model_dir += '/{}'.format(norm)
    model_path = '{}/{}.pt'.format(model_dir, model_name)
    model_dicts = all_models[norm]
    if not isinstance(model_dicts[model_name]['gdrive_id'], list):
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.isfile(model_path):
            download_gdrive(model_dicts[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
        
        model.load_state_dict(state_dict, strict=False)
        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for i, gid in enumerate(model_dicts[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i), map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
            except:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict, strict=False)
            model.models[i].eval()
        return model.eval()


def clean_accuracy(model, x, y, batch_size=100, device=None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def list_available_models(norm='Linf'):
    from model_zoo_old.models import model_dicts
    models = model_dicts[norm].keys()

    json_dicts = []
    for model_name in models:
        with open('./model_info/{}/{}.json'.format(norm, model_name), 'r') as model_info:
            json_dict = json.load(model_info)
        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict['venue'] == '' else json_dict['venue']
        json_dict['AA'] = float(json_dict['AA']) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -d['AA'])
    print('| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |')
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        if json_dict['model_name'] == 'Chen2020Adversarial':
            json_dict['architecture'] = json_dict['architecture'] + ' <br/> (3x ensemble)'
        if json_dict['model_name'] != 'Natural':
            print('| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'.format(
                i+1, json_dict['model_name'], json_dict['name'], json_dict['link'], json_dict['clean_acc'], json_dict['AA'],
                json_dict['architecture'], json_dict['venue']))
        else:
            print('| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*{}*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'.format(
                i + 1, json_dict['model_name'], json_dict['name'], json_dict['clean_acc'],
                json_dict['AA'], json_dict['architecture'], json_dict['venue']))

