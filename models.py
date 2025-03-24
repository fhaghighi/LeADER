import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x

class SimpleWrapper(nn.Module):
    def __init__(self, backbone, disease_head,anatomy_head):
        super(SimpleWrapper, self).__init__()
        if hasattr(backbone, 'fc'):
            print("backbone had fc, removed it.")
            backbone.fc = nn.Identity()
        if hasattr(backbone, 'head'):
            print("backbone had head, removed it.")
            backbone.head = nn.Identity()
        self.backbone = backbone
        self.disease_head = disease_head
        self.anatomy_head = anatomy_head

    def forward(self, x):
        output = self.backbone(x)
        disease_embeddings = self.disease_head(output)
        anatomy_embeddings = self.anatomy_head(output)
        return disease_embeddings,anatomy_embeddings


class AnatomyModelWrapper(nn.Module):
    def __init__(self, args):
        super(AnatomyModelWrapper, self).__init__()
        self.base_model = torchvision_models.__dict__[args.anatomy_model_arch]()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.anatomy_model_use_head=args.anatomy_model_use_head
        dim_embed = 1376
        latent = 2048
        self.fc1 = nn.Sequential(nn.Linear(latent, dim_embed), nn.BatchNorm1d(dim_embed), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(dim_embed, dim_embed), nn.BatchNorm1d(dim_embed))

    def forward(self, xb):
        x = self.base_model.maxpool(self.base_model.relu(self.base_model.bn1(self.base_model.conv1(xb))))
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.anatomy_model_use_head:
            x = self.fc1(x)
            x = self.fc2(x)
        x = nn.functional.normalize(x, dim=1)
        return x
