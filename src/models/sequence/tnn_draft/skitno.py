from  dataclasses import dataclass
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import logging_info
from .dpb_v4 import SimpleRMSNorm

from laxtnn.modules.skitno_inv_time import SKITnoInvTime 

@dataclass
class SkiTnoConfig:
    h: int
    #n: int
    dim: int
    #rpe_dim: int
    causal: bool
    #use_pad: bool
    #act: str
    #rpe_type: int
    #layers: int
    gamma: float
    #bias: bool
    act_type: str


class SKITNO(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        bias=True,
        # add
        index=0,
        act_fun="silu",
        causal=False,
        tno_expand_ratio=2,
        shrink_ratio=1,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="layernorm",
        # Toeplizt
        normalize=False,
        gamma=0.999,
        # token shift
        token_shift_type=-1,
        #ski
        rank=32,
        nk=16,
        **unused
    ):
        # add
        self.index = index

        super().__init__()
        self.d_output = d_model
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads

        self.expand_ratio = tno_expand_ratio
        self.resi_param = resi_param
        logging_info(f"self.expand_ratio {self.expand_ratio}")
        logging_info(f"self.resi_param {self.resi_param}")
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))
        d1 = int(self.expand_ratio * d_model)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = d_model
        self.head_dim = d1 // n_heads
        # d^2
        self.v_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.u_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.o = nn.Linear(d1, d_model, bias=bias)

        self.causal = causal
        self.act = self.get_act_fun(act_fun)
        logging_info(f"act_fun {act_fun}")
        logging_info(f"causal {self.causal}")

        # toep
        self.normalize = normalize
        self.gamma = gamma
        self.bias = bias
        config = SkiTnoConfig(
            h=self.num_heads,
            dim=self.head_dim,
            causal=self.causal,
            gamma=self.gamma,
            #jmei
            act_type="none",
        ).__dict__

        self.rank = rank
        self.nk = nk
        self.toep = SKITnoInvTime(r=rank, nk=nk, **config)

        logging_info(f"self.num_heads {self.num_heads}")
        logging_info(f"self.normalize {self.normalize}")
        logging_info(f"self.gamma {self.gamma}")
        logging_info(f"bias {bias}")

        # norm
        self.norm_type = norm_type
        self.pre_norm = self.get_norm_fun(self.norm_type, d2)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = self.get_norm_fun(norm_type, d1)
        logging_info(f"use_norm {self.use_norm}")
        logging_info(f"norm_type {self.norm_type}")

        self.token_shift_type = token_shift_type
        logging_info(f"self.token_shift_type {self.token_shift_type}")
        if self.token_shift_type == 1:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
        elif self.token_shift_type == 2:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.coef = 0.5

        self.par_init()

    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)

    def get_norm_fun(self, norm_type, d_model):
        if norm_type == "simplermsnorm":
            logging_info("here! simple rmsnorm")
            return SimpleRMSNorm(d_model)
        else:
            logging_info("here! layer norm")
            return nn.LayerNorm(d_model)
            # return nn.BatchNorm1d(d_model)

    def get_act_fun(self, act_fun):
        logging_info(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return torch.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":

            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)

            return f
        elif act_fun == "1+elu":

            def f(x):
                return 1 + F.elu(x)

            return f
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":

            def f(x):
                return torch.square(torch.relu(x))

            return f
        else:
            return lambda x: x

    def forward(self, x, state=None):
        # x: b, h * w, d
        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))  # (b, n, hd)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output)

        return output, None
