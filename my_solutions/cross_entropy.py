import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def cross_entropy_loss(logits, targets):

    # softmaxed = torch.softmax(logits, dim=-1)
    # B = logits.shape[0]
    # probs = softmaxed[torch.arange(B), targets]
    # return -torch.log(probs).mean()

    #pass  # log_probs = logits - logsumexp(...)


    # lets solve it including the softmax
    max_per_dim = torch.max(logits, dim=-1, keepdim=True).values
    safe_negative_logits = torch.sub(logits, max_per_dim)
    exp_logits = torch.exp(safe_negative_logits)
    #divide by sum accross dim
    sum_per_dim = torch.sum(exp_logits, dim=-1, keepdim=True)
    softmaxed = torch.div(exp_logits, sum_per_dim)

    # next step, we need to get the -log(liklihood) of each target
    target_indicies = [torch.arange(logits.shape[0]), targets] # this includes the batch, target indice
    liklihood = softmaxed[target_indicies]
    n_log_likelihood = torch.negative(torch.log(liklihood))

    #mean loss
    mean_loss = n_log_likelihood.mean()
    return mean_loss

