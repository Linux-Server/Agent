from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embed:int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.n_head = config.n_head
        self.n_embd = config.n_embed
        self.head_size = config.n_embed // config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,self.head_size).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_size).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_size).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
           dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)))
            
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
    def forward(self, idx):
        B,T = idx.size()
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(torch.arange(T, device=idx.device))
        x = token_embeddings + position_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

        
    @classmethod
    def from_pretrained(cls, model_name):
        "Loads the model from HuggingFace and returns an instance of the GPT class"
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name)
        config = GPTConfig(
            block_size=model.config.n_positions,
            vocab_size=model.config.vocab_size,
            n_layer=model.config.n_layer,
            n_head=model.config.n_head,
            n_embed=model.config.n_embd
        )
        gpt = cls(config)
        gpt.transformer.wte.weight.data.copy_(model.transformer.wte.weight.data)
        gpt.transformer.wpe.weight.data.copy_(model.transformer.wpe.weight.data)
        for i in range(config.n_layer):
            gpt.transformer.h[i].ln_1.weight.data.copy_(model.transformer.h[i].ln_1.weight.data)
            gpt.transformer.h[i].ln_1.bias.data.copy_(model.transformer.h[i].ln_1.bias.data)
            gpt.transformer.h[i].attn.c_attn.weight.data.copy_(model.transformer.h[i].attn.c_attn.weight.data.t())
            gpt.transformer.h[i].attn.c_attn.bias.data.copy_(model.transformer.h[i].attn.c_attn.bias.data)
            gpt.transformer.h[i].attn.c_proj.weight.data.copy_(model.transformer.h[i].attn.c_proj.weight.data.t())
            gpt.transformer.h[i].attn.c_proj.bias.data.copy_(model.transformer.h[i].attn.c_proj.bias.data)
            gpt.transformer.h[i].ln_2.weight.data.copy_(model.transformer.h[i].ln_2.weight.data)
            gpt.transformer.h[i].ln_2.bias.data.copy_(model.transformer.h[i].ln_2.bias.data)
            gpt.transformer.h[i].mlp.c_fc.weight.data.copy_(model.transformer.h[i].mlp.c_fc.weight.data.t())
            gpt.transformer.h[i].mlp.c_fc.bias.data.copy_(model.transformer.h[i].mlp.c_fc.bias.data)
            gpt.transformer.h[i].mlp.c_proj.weight.data.copy_(model.transformer.h[i].mlp.c_proj.weight.data.t())
            gpt.transformer.h[i].mlp.c_proj.bias.data.copy_(model.transformer.h[i].mlp.c_proj.bias.data)
        gpt.transformer.ln_f.weight.data.copy_(model.transformer.ln_f.weight.data)
        gpt.transformer.ln_f.bias.data.copy_(model.transformer.ln_f.bias.data)
        gpt.lm_head.weight.data.copy_(model.lm_head.weight.data)
        return gpt

model = GPT.from_pretrained("gpt2")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens).unsqueeze(0)

while tokens.size(1) < 50:
    logits = model(tokens)
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    tokens = torch.cat((tokens, next_token), dim=1)
print(tokenizer.decode(tokens.squeeze().tolist()))


    
