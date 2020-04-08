"""
AllenNLP just uses
`PyTorch optimizers <https://pytorch.org/docs/master/optim.html>`_ ,
with a thin wrapper to allow registering them and instantiating them `from_params`.
The available optimizers are
* `"adadelta" <https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta>`_
* `"adagrad" <https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad>`_
* `"adam" <https://pytorch.org/docs/master/optim.html#torch.optim.Adam>`_
* `"adamw" <https://pytorch.org/docs/master/optim.html#torch.optim.AdamW>`_
* `"huggingface_adamw"
  <https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW>`_
* `"sparse_adam" <https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam>`_
* `"sgd" <https://pytorch.org/docs/master/optim.html#torch.optim.SGD>`_
* `"rmsprop <https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop>`_
* `"adamax <https://pytorch.org/docs/master/optim.html#torch.optim.Adamax>`_
* `"averaged_sgd <https://pytorch.org/docs/master/optim.html#torch.optim.ASGD>`_
"""

import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class DenseSparseAdam(torch.optim.Optimizer):
    """
    NOTE: This class has been copied verbatim from the separate Dense and
    Sparse versions of Adam in Pytorch.
    Implements Adam algorithm with dense & sparse gradients.
    It has been proposed in Adam: A Method for Stochastic Optimization.
    Registered as an `Optimizer` with name "dense_sparse_adam".
    # Parameters
    params : `iterable`
        iterable of parameters to optimize or dicts defining parameter groups
    lr : `float`, optional (default: 1e-3)
        The learning rate.
    betas : `Tuple[float, float]`, optional (default: (0.9, 0.999))
        coefficients used for computing running averages of gradient
        and its square.
    eps : `float`, optional, (default: 1e-8)
        A term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        # Parameters
        closure : `callable`, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = (
                        grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    )
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group["eps"])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))

                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss