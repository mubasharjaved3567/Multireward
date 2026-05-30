import torch
import torch.nn.functional as F
from collections import defaultdict


# Layer groups — each reward routes to specific UNet layers
LAYER_GROUPS = {
    "cross_attn": lambda name: "attn2" in name,
    "self_attn":  lambda name: "attn1" in name,
    "conv":       lambda name: ("conv" in name and "attn" not in name and "norm" not in name),
    "resnets":    lambda name: "resnets" in name,
}

# Routing table — primary layers get full weight, secondary get SECONDARY_SCALE
# smooth_weights drives the reward proportions (trajectory-aware)
REWARD_ROUTING = {
    "r_align":      {"primary": ["cross_attn"], "secondary": ["self_attn"],         "idx": 0},
    "r_aesthetic":  {"primary": ["self_attn", "conv"], "secondary": ["cross_attn"], "idx": 1},
    "r_preference": {"primary": ["self_attn", "cross_attn"], "secondary": ["conv"], "idx": 2},
    "r_quality":    {"primary": ["resnets"], "secondary": ["conv"],                 "idx": 3},
}

SECONDARY_SCALE = 0.3


class LayerRouter:
    def __init__(self, unet):
        self.unet = unet
        self.group_params = defaultdict(list)
        for name, param in unet.named_parameters():
            for group, matcher in LAYER_GROUPS.items():
                if matcher(name):
                    self.group_params[group].append((name, param))
        print("[LayerRouter] Layer groups:")
        for g, params in self.group_params.items():
            n = sum(p.numel() for _, p in params)
            print(f"  {g:<14} {len(params):>4} tensors   {n:>12,} params")

    def _grad_norm(self, group):
        norms = [p.grad.float().norm().item()
                 for _, p in self.group_params[group]
                 if p.grad is not None]
        return sum(norms) / len(norms) if norms else 0.0

    def step(self, norm_scores, smooth_w, noise_pred, ref_pred,
             kl, opt, kl_beta=0.01, grad_clip=1.0):
        """
        Trajectory-aware routed backward pass.

        smooth_w: tensor of shape [4] from smooth_weights(reward_idx, T)
                  — drives HOW MUCH each reward contributes at this timestep
        layer routing — drives WHERE each reward's gradient flows in UNet

        Flow:
          1. composite_R = sum(norm_score[i] * smooth_w[i])
          2. Single backward — graph freed immediately
          3. Scale each param's grad by its routing weight
             (primary layers get full smooth_w[i], secondary get 0.3x)
          4. Clip + optimizer step
        """
        opt.zero_grad()
        log         = {}
        composite_R = 0.0

        # smooth_w drives the reward proportions (trajectory-aware)
        # routing drives the layer-specific grad scaling
        param_weights = {}
        for reward_key, routing in REWARD_ROUTING.items():
            r_val   = norm_scores.get(reward_key, 0.0)
            sw_val  = smooth_w[routing["idx"]].item()   # trajectory weight

            # composite reward uses smooth_w — NOT static weight
            composite_R += r_val * sw_val

            # per-param routing weight = reward_value * smooth_weight
            for g in routing["primary"]:
                for name, param in self.group_params[g]:
                    param_weights[name] = param_weights.get(name, 0.0) + r_val * sw_val

            for g in routing["secondary"]:
                for name, param in self.group_params[g]:
                    param_weights[name] = param_weights.get(name, 0.0) + r_val * sw_val * SECONDARY_SCALE

        # Single composite loss — one backward, graph freed immediately
        log_prob = -0.5 * F.mse_loss(
            noise_pred.float(), ref_pred.float(), reduction="mean"
        )
        loss = -composite_R * log_prob + kl_beta * kl
        loss.backward()

        # Scale each param's gradient by its routing weight
        for name, param in self.unet.named_parameters():
            if param.grad is not None and name in param_weights:
                scale = param_weights[name] / (composite_R + 1e-8)
                param.grad.mul_(scale)

        # Log grad norms per group
        for g in LAYER_GROUPS:
            log[f"gnorm_{g}"] = round(self._grad_norm(g), 6)

        total_gnorm = torch.nn.utils.clip_grad_norm_(
            self.unet.parameters(), grad_clip
        ).item()
        opt.step()

        log["composite_R"] = round(composite_R, 4)
        log["total_gnorm"] = round(total_gnorm, 6)
        log["kl"]          = round(kl.item(), 6)
        return log