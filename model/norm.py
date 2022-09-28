import torch
import torch.nn as nn
import numpy as np

class GroupBatchNorm2d(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(GroupBatchNorm2d, self).__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.weight, self.bias = None, None
        self.register_buffer('running_mean', torch.zeros(num_groups))
        self.register_buffer('running_var', torch.ones(num_groups))

        self.reset_parameters()

        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def extra_repr(self):
        s = ('{num_groups}, {num_features}, eps={eps}'
             ', affine={affine}')
        return s.format(**self.__dict__)

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x_reshape = x.view(N, G, int(C/G), H, W)
        mean = x_reshape.mean([0,2,3,4])
        var = x_reshape.var([0,2,3,4])

        if self.training:
            self.running_mean.data.copy_((self.momentum * self.running_mean) + (1.0 - self.momentum) * mean)
            self.running_var.data.copy_((self.momentum * self.running_var) + (1.0 - self.momentum) * (N / (N - 1) * var))
        else:
            mean = self.running_mean.data
            var = self.running_var.data

        # change shape
        current_mean = mean.view([1, G, 1, 1, 1]).expand_as(x_reshape)
        current_var = var.view([1, G, 1, 1, 1]).expand_as(x_reshape)
        x_reshape = (x_reshape-current_mean) / (current_var+self.eps).sqrt()
        x = x_reshape.view(N,C,H,W)

        if self.affine:
            return x * self.weight + self.bias
        else:
            return x

class USNorm(nn.Module):
    def __init__(self, num_features, norm_list):
        super(USNorm, self).__init__()
        self.num_features = num_features
        self.norm_list = norm_list

        # define list of normalizations
        normalization = []
        for item in self.norm_list:
            if item == 'bn':
                normalization.append(nn.BatchNorm2d(num_features))
            elif item == 'in':
                normalization.append(nn.InstanceNorm2d(num_features))
            else:
                norm_type = item[:item.index('_')]
                num_group = int(item[item.index('_') + 1:])
                if 'gn' in norm_type:
                    if 'r' in norm_type:
                        if int(num_features/num_group) > num_features:
                            normalization.append(nn.GroupNorm(num_features, num_features))
                        else:
                            normalization.append(nn.GroupNorm(int(num_features/num_group), num_features))
                    else:
                        if int(num_features / num_group) > num_features:
                            normalization.append(nn.GroupNorm(num_features, num_features))
                        else:
                            normalization.append(nn.GroupNorm(num_group, num_features))
                elif 'gbn' in norm_type:
                    if 'r' in norm_type:
                        normalization.append(GroupBatchNorm2d(int(num_features/num_group), num_features))
                    else:
                        normalization.append(GroupBatchNorm2d(num_group, num_features))

        self.norm = nn.ModuleList(normalization)

        self.norm_type = None

    def set_norms(self, norm_type=None):
        self.norm_type = norm_type

    def set_norms_mixed(self):
        self.norm_type = np.random.choice(self.norm_list)

    def forward(self, input):
        if self.norm is None:
            assert self.norm is not None
        else:
            assert self.norm_type in self.norm_list
            idx = self.norm_list.index(self.norm_type)

        y = self.norm[idx](input)

        return y
