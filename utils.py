# import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from model.norm import USNorm

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True):
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        dataset_normalization = None

    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=cifar10_mean, std=cifar10_std)

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    else:
        print('Wrong dataset:', dataset)
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=worker,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=worker,
    )
    return train_loader, test_loader, dataset_normalization


# pgd attack
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            # attack all cases
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[:, :, :, :], upper_limit - X[:, :, :, :])
            delta.data = d

            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


# evaluate on clean images with single norm
def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


# evaluate on clean images with random norms
def evaluate_standard_random_norms(test_loader, model, args):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


# evaluate on adv images with single norm
def evaluate_pgd(test_loader, model, attack_iters, restarts, args):
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return pgd_loss/n, pgd_acc/n


# evaluate on adv images with random norms
def evaluate_pgd_random_norms(test_loader, model, attack_iters, restarts, args, num_round=3):
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            # random select a path to attack
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

            # random select a path to infer
            if args.mixed:
                set_random_norm_mixed(args, model)
            else:
                set_random_norm(args, model)

            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    return pgd_loss/n, pgd_acc/n


# random norm for entire network
def set_random_norm(args, model):
    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    norm = np.random.choice(norm_list)
    model.module.set_norms(norm)
    return norm


# random norm for each layer
def set_random_norm_mixed(args, model):
    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type, args.gn_type)
    # get norm module from model
    for name, module in model.named_modules():
        if isinstance(module, USNorm):
            norm = np.random.choice(norm_list)
            module.set_norms(norm)


# setup random space for norms
def set_norm_list(min_group, max_group, random_type, gn_type):
    num_group_list = []
    for i in range(min_group, max_group + 1):
        num_group_list.append(2 ** i)
    # define norm list
    norm_list = []
    if 'bn' in random_type:
        norm_list.append('bn')
    if 'in' in random_type:
        norm_list.append('in')
    if '_' not in gn_type:
        for item in num_group_list:
            norm_list.append(gn_type + '_' + str(item))
    else:
        gn_str = gn_type[:gn_type.index('_')]
        gbn_str = gn_type[gn_type.index('_')+1:]
        for item in num_group_list:
            norm_list.append(gn_str + '_' + str(item))
            norm_list.append(gbn_str + '_' + str(item))

    return norm_list
