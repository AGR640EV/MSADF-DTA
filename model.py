# MSADF_DTA.py
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from collections import OrderedDict
class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError(f'expected 2D input (got {input.dim()}D)')

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = self.momentum if self.momentum is not None else 0.0
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps
        )

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        data.x = F.relu(self.norm(self.conv(x, edge_index)))
        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated = torch.cat(data.x, dim=1)
        data.x = concated
        data = self.conv1(data)
        return data

    def forward(self, data):
        if isinstance(data.x, torch.Tensor):
            data.x = [data.x]
        data = self.bn_function(data)
        data = self.conv2(data)
        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module(f'layer{i+1}', layer)

    def forward(self, data):
        features = [data.x]
        for _, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features
        data.x = torch.cat(data.x, dim=1)
        return data
# atomic_graph_encoder
class AtomicGraphEncoder(nn.Module):
    def __init__(self, num_input_features=22, out_dim=96, growth_rate=32,
                 block_config=(8, 8), bn_sizes=(2, 2)):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_input_features, growth_rate, bn_sizes[i])
            self.features.add_module(f'block{i+1}', block)
            num_input_features += num_layers * growth_rate
            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module(f'transition{i+1}', trans)
            num_input_features = num_input_features // 2
        self.classifier = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        for layer in self.features:
            data = layer(data)
        x = global_mean_pool(data.x, data.batch)
        x = self.classifier(x)
        return x

# brics_graph_encoder
class BRICSGrapgEncoder(nn.Module):
    def __init__(self, num_input_features=768, out_dim=96, growth_rate=32):
        super().__init__()
        # BRICS 图相对浅
        block_config = (4, 4)
        bn_sizes = (1, 1)
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_input_features, growth_rate, bn_sizes[i])
            self.features.add_module(f'block{i+1}', block)
            num_input_features += num_layers * growth_rate
            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module(f'transition{i+1}', trans)
            num_input_features = num_input_features // 2
        self.classifier = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        for layer in self.features:
            data = layer(data)
        x = global_mean_pool(data.x, data.batch)
        x = self.classifier(x)
        return x

# protein_encoder
class Conv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
    def forward(self, x):
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size, stride, padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module(f'conv_layer{layer_idx+1}', Conv1dReLU(out_channels, out_channels, kernel_size, stride, padding))
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)

class ProteinEncoder(nn.Module):
    def __init__(self, block_num=3, vocab_size=21, embedding_num=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList([StackCNN(i + 1, embedding_num, 96, 3, padding=1) for i in range(block_num)])
        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, dim=-1)
        x = self.linear(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, feat1, feat2):
        gate_weight = self.gate(torch.cat([feat1, feat2], dim=-1))
        gated_feat = gate_weight * feat1 + (1 - gate_weight) * feat2
        feat_diff = feat1 - feat2
        fused_feat = self.fusion(torch.cat([gated_feat, feat_diff], dim=-1))
        return fused_feat

class GatedFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, feat1, feat2):
        gate_weight = self.gate(torch.cat([feat1, feat2], dim=-1))
        gated_feat = gate_weight * feat1 + (1 - gate_weight) * feat2
        feat_diff = feat1 - feat2
        fused_feat = self.fusion(torch.cat([gated_feat, feat_diff], dim=-1))
        return fused_feat    


class MSADF_DTA(nn.Module):
    def __init__(self, block_num=3, vocab_protein_size=21,
                 embedding_size=128, filter_num=32, out_dim=1):
        super().__init__()
        self.protein_encoder = ProteinEncoder(block_num, vocab_protein_size, embedding_size)
        self.atomic_encoder = AtomicGraphEncoder(out_dim=filter_num * 3)
        self.brics_encoder = BRICSGrapgEncoder(out_dim=filter_num * 3)
        self.graph_fusion = GatedFusion(feat_dim=filter_num * 3)
        self.classifier = nn.Sequential(
            nn.Linear(96 + filter_num * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, atomic_batch, brics_batch, target_batch):
        protein_feat = self.protein_encoder(target_batch)
        atomic_feat = self.atomic_encoder(atomic_batch)
        brics_feat = self.brics_encoder(brics_batch)

        fused_graph_feat = self.graph_fusion(atomic_feat, brics_feat)
        combined_feat = torch.cat([protein_feat, fused_graph_feat], dim=-1)
        out = self.classifier(combined_feat)
        return out

