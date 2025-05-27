from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def process_sequences(sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
    attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
    seq_length = attention_mask.size(1)

    # The following code is equivalent to:
    #
    # for i in range(attention_mask.size(0)):
    #     for t in reversed(range(seq_length)):
    #         if attention_mask[i][t] > 0.5:
    #             attention_mask[i][min(t + 1, seq_length - 1)] = True
    #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
    #             break
    #
    eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
    sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

    # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
    first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
    mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
    attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

    # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
    state_seq = sequences[:, input_len - 1 : -1]
    action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
    action_mask[:, 0] = 1

    return sequences, attention_mask, action_mask


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
    process_rewards: Union[torch.Tensor, list[torch.Tensor]] = None,
    prime_coef: float = 1.0,
    verifier_coef: float = 1.0,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    # if action_mask is not None:
    if isinstance(kl, list):
        # TODO: write a more efficient version
        # TODO: use action_mask to find the last critic token
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i] * verifier_coef
            
            # TODO: Maybe we need to do reward shaping on process rewards
            if process_rewards is not None:
                kl_reward += process_rewards[i] * prime_coef
            
            reward.append(kl_reward)
    else:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype)) * verifier_coef

        reward = last_reward + kl_reward

        # TODO: Maybe we need to do reward shaping on process rewards
        if process_rewards is not None:
            reward += process_rewards * prime_coef

    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def log_probs_and_entropy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))

    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return log_probs_labels.squeeze(-1), entropy


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values
