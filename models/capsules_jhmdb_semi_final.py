import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.pytorch_i3d import InceptionI3d
from collections import deque


# PrimaryCaps(832, 8, 9, P, stride=1)
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """

    # def __init__(self, A=32, B=64, K=9, P=4, stride=1):
    def __init__(self, A, B, K, P, stride):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                              kernel_size=K, stride=stride, bias=True)
        self.pose.weight.data.normal_(0.0, 0.1)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                           kernel_size=K, stride=stride, bias=True)
        self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


# ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """

    def __init__(self, B, C, K, P, stride, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        self._lambda = 1e-6
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2 * math.pi))
        # self.ln_2pi = torch.cuda.HalfTensor(1).fill_(math.log(2*math.pi))

        # params
        # Note that \beta_u and \beta_a are per capsul/home/bruce/projects/capsulese type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.randn(C, self.psize))
        self.beta_a = nn.Parameter(torch.randn(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K[0] * K[1] * B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        cost_h = cost_h.sum(dim=2)

        cost_h_mean = torch.mean(cost_h, dim=1, keepdim=True)

        cost_h_stdv = torch.sqrt(torch.sum(cost_h - cost_h_mean, dim=1, keepdim=True) ** 2 / C + eps)
        # self._lambda = 1e-03
        # a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))

        # cost_h_mean = cost_h_mean.sum(dim=2)
        # cost_h_stdv = cost_h_stdv.sum(dim=2)

        a_out = self.sigmoid(self._lambda * (self.beta_a - (cost_h_mean - cost_h) / (cost_h_stdv + eps)))

        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu) ** 2 / (2 * sigma_sq) \
                   - torch.log(sigma_sq.sqrt()) \
                   - 0.5 * self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(eps + a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1. / C)
        # r = torch.cuda.HalfTensor(b, B, C).fill_(1./C)
        # print(r.dtype)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B * (psize + 1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                 for h_idx in range(0, h - K + 1, stride)] \
                for k_idx in range(0, K)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def add_pathes2(self, x, B, K=(3, 3), psize=4, stride=(1, 1)):
        b, h, w, c = x.shape
        assert c == B * (psize + 1)

        oh = int((h - K[0] + 1) / stride[0])
        ow = int((w - K[1] + 1) / stride[1])

        idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - K[0] + 1, stride[0])] for k_idx in range(0, K[0])]
        idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - K[1] + 1, stride[1])] for k_idx in range(0, K[1])]

        x = x[:, idxs_h, :, :]
        x = x[:, :, :, idxs_w, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P * P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P * P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)

        # coor_h = torch.cuda.HalfTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        # coor_w = torch.cuda.HalfTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h * w * B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            # x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)
            x, oh, ow = self.add_pathes2(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B * self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B * self.psize:].contiguous()

            p_in = p_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, 1)
            # p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            # a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C * self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            # print(p_out.shape, a_out.shape)
            # print('conv cap activations',a_out[0].sum().item(),a_out[0].size())
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B * (self.psize + 1)
            assert 1 == self.K[0] and 1 == self.K[1]
            assert 1 == self.stride[0] and 1 == self.stride[1]
            # assert 1 == self.K
            # assert 1 == self.stride
            p_in = x[:, :, :, :self.B * self.psize].contiguous()
            p_in = p_in.view(b, h * w * self.B, self.psize)
            a_in = x[:, :, :, self.B * self.psize:].contiguous()
            a_in = a_in.view(b, h * w * self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


class MultiLayerDiffCrossAttention(nn.Module):
    def __init__(self, channels, num_layers=2, eps=1e-8, fusion_mode='diff_cross_attention'):
        super().__init__()
        self.num_layers = num_layers
        self.eps = eps
        self.scale = channels ** -0.5
        self.fusion_mode = fusion_mode

        # Per-layer query and dual key/value projections.
        self.q_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])
        self.k1_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])
        self.v1_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])
        self.k2_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])
        self.v2_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])

        # Output projections following the standard Transformer block design.
        self.out_projs = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_layers)
        ])

        # Gating network controlling the suppression strength of A2 (lambda).
        # Input: q_global + k1_global; output: lambda in [0, 1].
        self.gate_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * 2, max(channels // 8, 32)),
                nn.Tanh(),
                nn.Linear(max(channels // 8, 32), 1),
                nn.Sigmoid()  # 0 keeps A2 unchanged; 1 fully subtracts A2.
            ) for _ in range(num_layers)
        ])

        # Value fusion gate: v = gate_v * v1 + (1 - gate_v) * v2.
        self.v_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])

        # Layer normalization in a pre-LN layout.
        self.ln_q = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(num_layers)
        ])
        self.ln_kv1 = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(num_layers)
        ])
        self.ln_kv2 = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(num_layers)
        ])

        # Initialization.
        self._init_weights()

    def _init_weights(self):
        """Initialize attention and gating weights."""
        for proj_list in [self.q_projs, self.k1_projs, self.v1_projs,
                          self.k2_projs, self.v2_projs, self.out_projs]:
            for proj in proj_list:
                nn.init.normal_(proj.weight, mean=0.0, std=0.02)
                nn.init.zeros_(proj.bias)

        for gate_proj in self.gate_projs:
            for layer in gate_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)
            # Use a negative bias so the initial lambda remains small.
            last_linear = gate_proj[-2]
            nn.init.constant_(last_linear.bias, -1.0)  # Initial lambda is approximately 0.26.

        for v_gate in self.v_gate:
            nn.init.normal_(v_gate[0].weight, std=0.02)
            nn.init.zeros_(v_gate[0].bias)

    def stable_softmax(self, x, dim=-1):
        """Numerically stable softmax."""
        x = x - x.max(dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x)
        return exp_x / (exp_x.sum(dim=dim, keepdim=True) + self.eps)

    def forward(self, x_q, x_kv):
        """
        Differential cross-attention reconstructs x_q while suppressing common-mode noise in x_kv.
        Args:
            x_q: [1, L, C] query sequence to be enhanced.
            x_kv: [N, L, C] key/value reference sequences containing noisy samples.

        Returns:
            out: [1, L, C] denoised/reconstructed query sequence.
        """
        assert x_q.size(0) == 1, "x_q must contain a single query sample."
        assert x_kv.size(0) >= 2, "x_kv needs at least two reference sequences for the differential term."

        out = x_q  # Initial input.
        N, L, C = x_kv.size()

        for i in range(self.num_layers):
            residual = out
            # Pre-LN
            out = self.ln_q[i](out)

            # Linear projections.
            q = self.q_projs[i](out)                    # [1, L, C]
            k1 = self.k1_projs[i](x_kv)                 # [N, L, C]
            v1 = self.v1_projs[i](x_kv)
            k2 = self.k2_projs[i](x_kv)
            v2 = self.v2_projs[i](x_kv)

            # Normalize features to enhance differential contrast.
            k1 = self.ln_kv1[i](k1)
            k2 = self.ln_kv2[i](k2)

            # Remove the batch dimension and transpose [N, L, C] to [L, N, C].
            q = q.squeeze(0)          # [L, C]
            k1 = k1.transpose(0, 1)   # [L, N, C]
            v1 = v1.transpose(0, 1)
            k2 = k2.transpose(0, 1)
            v2 = v2.transpose(0, 1)

            # Compute attention logits.
            A1_logits = torch.einsum('l c, l n c -> l n', q, k1) * self.scale  # [L, N]
            A2_logits = torch.einsum('l c, l n c -> l n', q, k2) * self.scale

            # Differential attention: A_diff = A1 - lambda * A2.
            q_global = q.mean(dim=0, keepdim=True)      # [1, C]
            k1_global = k1.mean(dim=0)                  # [N, C]
            q_exp = q_global.expand(N, -1)

            if self.fusion_mode == 'linear':
                linear_logits = torch.einsum('c,nc->n', q_global.squeeze(0), k1_global) * self.scale
                linear_weights = self.stable_softmax(linear_logits, dim=0)
                linear_weights = linear_weights.unsqueeze(0).expand(L, -1)
                updated = torch.einsum('l n, l n c -> l c', linear_weights, v1)
            elif self.fusion_mode == 'cross_attention':
                attn = self.stable_softmax(A1_logits, dim=-1)
                updated = torch.einsum('l n, l n c -> l c', attn, v1)
            else:
                gate_input = torch.cat([q_exp, k1_global], dim=-1)  # [N, 2C]
                lambda_gate = self.gate_projs[i](gate_input)        # [N, 1]
                lambda_expand = lambda_gate.t().expand(L, -1)       # [L, N]

                raw_diff = A1_logits - lambda_expand * A2_logits    # [L, N]
                A_diff = self.stable_softmax(raw_diff, dim=-1)      # [L, N]

                v_gate_input = torch.cat([q_exp, k1_global], dim=-1)
                v_weight = self.v_gate[i](v_gate_input)               # [N, C]
                v_weight = v_weight.unsqueeze(0).expand(L, -1, -1)    # [L, N, C]
                v_fused = v_weight * v1 + (1 - v_weight) * v2         # [L, N, C]
                updated = torch.einsum('l n, l n c -> l c', A_diff, v_fused)
            updated = updated.unsqueeze(0)  # [1, L, C]

            # Output projection.
            updated = self.out_projs[i](updated)

            # Residual connection.
            out = residual + updated  # [1, L, C]

        return out

class CapsNet(nn.Module):

    def __init__(self, pt_path='path/to/I3D_weights.pth', P=4, pretrained_load='i3d',
                 queue_size=4, num_attention_layers=2, fusion_mode='diff_cross_attention'):
        super(CapsNet, self).__init__()
        self.P = P

        self.conv1 = InceptionI3d(157, in_channels=3, final_endpoint='Mixed_4f')
        pretrained_weights = torch.load(pt_path)
        weights = self.conv1.state_dict()
        loaded_layers = 0
        # print(len(pretrained_weights.keys()), len(weights.keys()))
        # print(weights.keys())
        # print("**********************************************")
        # print(pretrained_weights.keys())
        for name in weights.keys():
            if name in pretrained_weights.keys():
                weights[name] = pretrained_weights[name]
                loaded_layers += 1

        self.conv1.load_state_dict(weights)
        print("Loaded I3D pretrained weights from ", pt_path, " for layers: ", loaded_layers)

        self.primary_caps = PrimaryCaps(832, 32, 9, P, stride=1)
        #self.conv_caps = ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
        self.conv_caps = ConvCaps(32, 21, (1, 1), P, stride=(1, 1), iters=3)

        #self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1 = nn.ConvTranspose2d(336, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1.weight.data.normal_(0.0, 0.02)

        self.upsample2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1,
                                            output_padding=1)
        self.upsample2.weight.data.normal_(0.0, 0.02)

        # self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=1,output_padding=(0,1,1))
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1,
                                            output_padding=1)
        self.upsample3.weight.data.normal_(0.0, 0.02)

        self.upsample4 = nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1,
                                            output_padding=(1, 1, 1))
        self.upsample4.weight.data.normal_(0.0, 0.02)

        self.dropout3d = nn.Dropout3d(0.5)

        self.smooth = nn.ConvTranspose3d(128, 1, kernel_size=3, padding=1)
        self.smooth.weight.data.normal_(0.0, 0.02)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.conv28 = nn.Conv2d(832, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv56 = nn.Conv3d(192, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv112 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.queue_size =  queue_size  # Maximum queue length for each class.
        self.num_classes = 21  # Number of JHMDB action classes.
        self.class_queues = {i: deque(maxlen=self.queue_size) for i in range(self.num_classes)}
        self.cross_attn = MultiLayerDiffCrossAttention(
            channels=544,
            num_layers=num_attention_layers,
            fusion_mode=fusion_mode,
        )

    def load_pretrained_weights(self):
        saved_weights = torch.load('./savedweights/weights_referit')
        self.load_state_dict(saved_weights, strict=False)
        print('loaded referit pretrained weights for whole network')

    def load_previous_weights(self, weightfile):
        saved_weights = torch.load(weightfile)
        self.load_state_dict(saved_weights, strict=False)
        print('loaded weights from previous run: ', weightfile)

    def caps_reorder(self, imgcaps):
        h = imgcaps.size()[1]
        w = imgcaps.size()[2]
        img_data = imgcaps.size()[3]
        num_imgcaps = int(img_data / (self.P * self.P))

        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, img_acts), dim=-1)
        return combined_caps
    def update_feature_queue(self, x, mask, preds):
        """
        Update each class-wise feature queue with PrimaryCaps features.
        
        Args:
            x: Tensor, PrimaryCaps output [B, H, W, C].
            mask: Tensor indicating whether each sample is labeled [B]; 0 means unlabeled.
        """
        B, H, W, C = x.shape
        for i in range(B):
            if not mask[i]:  # Store features only for unlabeled samples.
                class_id = preds[i].item()  # Predicted class for the current sample.
                sample_feat = x[i].detach().cpu()  # [H, W, C]
                self.class_queues[class_id].append(sample_feat)


    def all_queues_full(self):
        """
        Return True when all class-wise feature queues are full.
        """
        for class_id, queue in self.class_queues.items():
            if len(queue) < self.queue_size:
                # print(f"Class {class_id} queue not full: {len(queue)}/{self.queue_size}")
                return False
        # print("All class queues are full.")
        return True


    def forward(self, img, classification, concat_labels, epoch, thresh_ep, recon_start_epoch, is_teacher=False,ema_model=None):
        '''
        INPUTS:
        img is of shape (B, 3, T, H, W) - B is batch size, T is number of frames (8 in our experiments), H and W are the height and width of frames (224x224 in our experiments)
        classification is of shape (B, ) - B is batch size - this contains the ground-truth class which will be used for masking at training time
        
        OUTPUTS:
        out is a list of segmentation masks (all copies of on another) of shape (B, T, H, W) - B is batch size, T is number of frames (8 in our experiments), H and W is the heights and widths (224x224 in our experiments)
        actor_prediction is the actor prediction (B, C) - B is batch size, C is the number of classes
        
        '''

        x, cross56, cross112 = self.conv1(img)

        # For 3d Dropout
        x = self.dropout3d(x)

        x = x.view(-1, 832, 28, 28)
        cross28 = x.clone()
        x = self.primary_caps(x)

        x = self.caps_reorder(x)
        x_clone = x.clone()
        combined_caps = self.conv_caps(x)

        h = combined_caps.size()[1]
        w = combined_caps.size()[2]
        caps = int(combined_caps.size()[3] / ((self.P * self.P) + 1))
        ranges = int(caps * self.P * self.P)
        activations = combined_caps[:, :, :, ranges:ranges + caps]
        poses = combined_caps[:, :, :, :ranges]

        # clone to assign different id blocks for each
        actor_prediction = activations.clone()
        feat_shape = activations.clone()

        feat_shape = torch.reshape(feat_shape, (
        feat_shape.shape[0], feat_shape.shape[1] * feat_shape.shape[2], feat_shape.shape[3]))
        
        actor_prediction = torch.mean(actor_prediction, 1)
        actor_prediction = torch.mean(actor_prediction, 1)
        poses = poses.view(-1, h, w, caps, self.P * self.P)

        
        if self.training:
            activations_labeled = torch.eye(caps).to('cuda')[classification.long()]
            activations_labeled = torch.squeeze(activations_labeled, 1)

            # predicted action
            # epoch value begins from 1
            if epoch < thresh_ep:
                # give equal weightage to each class before thresh epoch
                # model is not sure
                activations_unlabeled = torch.ones_like(activations_labeled)
            else:
                # pick the class with highest pred since model is confident now
                activations_unlabeled = torch.eye(caps).to('cuda')[torch.argmax(actor_prediction, dim=1)]
            # print(activations_labeled.shape, activations_unlabeled.shape)
            activations = [activations_unlabeled[act] if concat_labels[act] == 0 else activations_labeled[act] for act
                           in range(len(concat_labels))]
            activations = torch.stack(activations)
            # print(activations_labeled, activations_unlabeled)
            
            activations = activations.view(-1, caps, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations.cuda()

        else:
            activations = torch.eye(caps).to('cuda')[torch.argmax(actor_prediction, dim=1)]
            activations = activations.view(-1, caps, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = torch.unsqueeze(activations, 1)
            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations.cuda()

        
        
        poses = poses * activations
        poses = poses.view(-1, h, w, ranges)
        poses = poses.permute(0, 3, 1, 2)
        x = poses
        B, C, H, W = poses.shape        
        x = self.relu(self.upsample1(x))
        x = x.view(-1, 64, 1, 28, 28)

        cross28_main = cross28.view(-1, 832, 28, 28)
        cross28_main = self.relu(self.conv28(cross28_main))
        cross28_main = cross28_main.view(-1, 64, 1, 28, 28)
        x = torch.cat((x, cross28_main), dim=1)

        x = self.relu(self.upsample2(x))
        cross56_main = self.relu(self.conv56(cross56))
        x = torch.cat((x, cross56_main), dim=1)

        x = self.relu(self.upsample3(x))
        cross112_main = self.relu(self.conv112(cross112))
        x = torch.cat((x, cross112_main), dim=1)

        x = self.upsample4(x)
        # Optional fallback for CUDA memory pressure.
        # with torch.backends.cudnn.flags(enabled=False):
        #     x = self.upsample4(x)
        x = self.dropout3d(x)
        x = self.smooth(x)
        out = x.view(-1, 1, 8, 224, 224)
        
        
        concat_labels = concat_labels.bool()
        labeled_idx = concat_labels.nonzero(as_tuple=True)[0]
        unlabeled_idx = (~concat_labels).nonzero(as_tuple=True)[0]
        preds = torch.argmax(actor_prediction.detach(), dim=1)
        # # Clone for reconstruction branch
        if is_teacher:
            self.update_feature_queue(x_clone, concat_labels, preds)
            return out, actor_prediction, feat_shape, None
        # --- Reconstruction branch ---
        elif not is_teacher and epoch < recon_start_epoch:
            return out, actor_prediction, feat_shape, None
        else:
            # if self.all_queues_full():
            #     print("All class queues are full.")

            if len(labeled_idx) > 0 and len(unlabeled_idx) > 0:
                reconstructed_feats = []
                classification_1 = classification.squeeze(1)
                # Reconstruct features for all labeled samples.
                for idx in labeled_idx:
                    label_class = int(classification_1[idx].item())
                    class_queue_feats = list(ema_model.class_queues[label_class])  # [queue_size, H, W, C]
                    query_feat = x_clone[idx].unsqueeze(0)  # [1, H, W, C]
                    if len(class_queue_feats) > 1:
                        # 1. Build attention inputs.
                        queue_feats = torch.stack(class_queue_feats, dim=0)  # [L_q, H, W, C]
                        L, H, W, C = queue_feats.shape
                        
                        # 2. Flatten spatial dimensions to [L_q, H*W, C] and [1, H*W, C].
                        queue_kv = queue_feats.view(L, H*W, C).to(query_feat.device)
                        query_q = query_feat.view(1, H*W, C)
                        
                        # 3. Fuse features with cross-attention.
                        fused = self.cross_attn(query_q, queue_kv)  # [1, H*W, C]
                        
                        # 4. Restore the original spatial layout [1, H, W, C].
                        fused_feat = fused.view(1, H, W, C)
                    else:
                        fused_feat = query_feat  # Keep [1, H, W, C].
                    reconstructed_feats.append(fused_feat.squeeze(0))   # [H, W, C]
                # Replace the labeled part with reconstructed features.
                x_recon_copy = x_clone.clone()
                x_recon_copy[labeled_idx] = torch.stack(reconstructed_feats, dim=0)
                x_recon = x_recon_copy  # Keep shape [B, H, W, C].
                combined_caps_recon = self.conv_caps(x_recon)

                activations_recon = combined_caps_recon[:, :, :, ranges:ranges + caps] #activations:(8, 20, 20, 24)
                actor_prediction_recon = activations_recon #actor_prediction:(8, 20, 20, 24)
                actor_prediction_recon = torch.mean(actor_prediction_recon, 1) #actor_prediction_in:(8, 20, 20, 24), actor_prediction_out:(8, 20, 24)
                actor_prediction_recon = torch.mean(actor_prediction_recon, 1) #actor_prediction_in:(8, 20, 24), actor_prediction_out:(8, 24)

                return out, actor_prediction, feat_shape, actor_prediction_recon


outputs = []
def hook(module, input, output):
    outputs.append(output)


if __name__ == '__main__':
    activation = {}

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = CapsNet(pretrained_load=True)
    model = model.to(device)
    # summary(model, [(3, 8, 224, 224), [1, 1]])
    fstack = torch.rand(4, 3, 8, 224, 224).to(device)
    actor = torch.ones(4).to(device)
    concat_labels = torch.tensor([0., 1., 1., 0.]).to(device)
    print(actor, concat_labels)
    # print(fstack.shape)
    out, ap, feat_shape = model(fstack, actor, concat_labels, 5, 8)
    print(out.shape)
