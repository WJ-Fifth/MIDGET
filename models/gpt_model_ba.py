"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from .resnet import Resnet1D


def top_select(logits_up, logits_down, up_codebook, down_codebook):
    probs_up = F.softmax(logits_up, dim=-1)
    probs_down = F.softmax(logits_down, dim=-1)
    # print("probs up", probs_up.shape)

    _, idx_up = torch.topk(probs_up, k=1, dim=-1)
    _, idx_down = torch.topk(probs_down, k=1, dim=-1)

    probs_up_codes = torch.matmul(probs_up, up_codebook.t())
    probs_down_codes = torch.matmul(probs_up, down_codebook.t())
    # print("up top select", ix_up.shape)
    return (probs_up_codes, probs_down_codes), (idx_up, idx_down)


class GPT_BA_Model(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.music_downsample = MusicDownSample(config)
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, shift=None, codebooks=None):
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down = xs
        for k in range(cond.size(1) // 8):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(
                    block_shift + (k - block_size - 1) % (block_size - block_shift + 1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(
                    block_shift + (k - block_size - 1) % (block_size - block_shift + 1)):]  # crop context if needed

            cond_input = cond[:, :(k + 1) * 8] if k < block_size else cond[:, (k - (
                    block_shift + (k - block_size - 1) % (block_size - block_shift + 1)) + 1) * 8: (k + 1) * 8]

            _, logits, _ = self.forward((x_cond_up, x_cond_down), cond_input, targets=None, codebooks=codebooks)
            # jj += 1
            # pluck the logits at the final step and scale by temperature
            logit_up, logit_down = logits
            ix_up = logit_up[:, -1, :]
            ix_down = logit_down[:, -1, :]
            # print("logit up", logit_up.shape)
            # print("logit down", logit_down.shape)
            #
            # probs_up = F.softmax(logit_up, dim=-1)
            # probs_down = F.softmax(logit_down, dim=-1)
            #
            # _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            # _, ix_down = torch.topk(probs_down, k=1, dim=-1)

            # append to the sequence and continue
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)

        return x_up.view(1, 1, -1), x_down.view(1, 1, -1)

    def forward(self, xs, cond, targets=None, codebooks=None):
        (x_up, x_down) = xs

        if codebooks is not None:
            (up_codebook, down_codebook) = codebooks
        else:
            (up_codebook, down_codebook) = None, None

        (targets_up, targets_down) = None, None
        if targets is not None:
            targets_up, targets_down = targets

        music_feature = self.music_downsample(cond)
        if music_feature.shape[0] > 1:
            input_music_feature = music_feature[:, :-1]
        else:
            input_music_feature = music_feature

        feat = self.gpt_base(x_up, x_down, input_music_feature)

        logits_up, logits_down, loss_up, loss_down = self.gpt_head(feat, targets)

        (probs_up_codes, probs_down_codes), (idx_up, idx_down) = \
            top_select(logits_up, logits_down, up_codebook, down_codebook)

        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down
        else:
            loss = None

        return (probs_up_codes, probs_down_codes), (idx_up, idx_down), loss


class MusicDownSample(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_channel = 438
        output_channel = 438
        down_rate = 3
        stride = 2
        width = 512
        depth = 3
        m_conv = 1.0

        blocks = []

        for i in range(down_rate):
            block = nn.Sequential(
                nn.Conv1d(in_channels=input_channel if i == 0 else width, out_channels=width, kernel_size=3,
                          stride=2, padding=1),
                Resnet1D(n_in=width, n_depth=depth, m_conv=m_conv,
                         dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False),
            )
            blocks.append(block)
        block = nn.Conv1d(in_channels=width, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
        blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x_in = self.preprocess(x)
        x_out = self.model(x_in)
        x_out = self.postprocess(x_out)

        return x_out

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x


class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        t = T // 3
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :t, :t].repeat(1, 1, 3, 3) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondGPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd)
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size * 3, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.block_size = config.block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, cond):
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # if self.requires_head:

        token_embeddings_up = self.tok_emb_up(idx_up)  # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down)  # each index maps to a (learnable) vector
        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings_up, token_embeddings_down], dim=1)

        position_embeddings = torch.cat(
            [self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size + t, :],
             self.pos_emb[:, self.block_size * 2:self.block_size * 2 + t, :]],
            dim=1)  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)

        return x


class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, targets=None):

        x = self.blocks(x)
        x = self.ln_f(x)
        N, T, C = x.size()
        t = T // 3
        logits_up = self.head_up(x[:, t:t * 2, :])
        logits_down = self.head_down(x[:, t * 2:t * 3, :])  # down half

        # if we are given some desired targets also calculate the loss
        loss_up, loss_down = None, None

        if targets is not None:
            targets_up, targets_down = targets

            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))

        return logits_up, logits_down, loss_up, loss_down
