"""A tiny real two-step LoRA optimization, restricted to CPU.

No pretrained model, dataset, GPU, download, or environment mutation.
It demonstrates why at standard LoRA initialization (B=0), the first step
updates B and the subsequent step can propagate gradient into A.
"""


def two_step_lora_demo():
    import torch

    torch.manual_seed(7)

    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(3, 2, bias=False)
            self.lora_A = torch.nn.Linear(3, 2, bias=False)
            self.lora_B = torch.nn.Linear(2, 2, bias=False)
            self.base.weight.requires_grad_(False)
            torch.nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            # r=2, alpha=2 so alpha/r = 1
            return self.base(x) + self.lora_B(self.lora_A(x))

    model = TinyLoRA().cpu()
    x = torch.tensor([[1., 0., 1.], [0., 1., -1.],
                      [1., 1., 0.], [-1., 0., 1.]])
    target = torch.tensor([[0.4, -0.2], [0.1, 0.5],
                           [-0.3, 0.2], [0.3, -0.4]])
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=0.1
    )
    base_before = model.base.weight.detach().clone()
    a_before = model.lora_A.weight.detach().clone()
    b_before = model.lora_B.weight.detach().clone()
    steps = []

    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        a_grad = float(model.lora_A.weight.grad.norm())
        b_grad = float(model.lora_B.weight.grad.norm())
        if not (torch.isfinite(loss).item() and torch.isfinite(model.lora_A.weight.grad).all().item()
                and torch.isfinite(model.lora_B.weight.grad).all().item()):
            raise RuntimeError("Non-finite gradient/loss")
        steps.append({
            "step": index + 1, "loss_before_update": float(loss.detach()),
            "A_grad_norm": a_grad, "B_grad_norm": b_grad
        })
        optimizer.step()

    frozen_ok = torch.equal(base_before, model.base.weight)
    a_changed = not torch.equal(a_before, model.lora_A.weight)
    b_changed = not torch.equal(b_before, model.lora_B.weight)
    if not (frozen_ok and a_changed and b_changed and steps[0]["A_grad_norm"] == 0
            and steps[0]["B_grad_norm"] > 0 and steps[1]["A_grad_norm"] > 0):
        raise RuntimeError("LoRA initialization / gradient audit failed")

    return {
        "kind": "CPU-only two-step teaching example, NOT SmolVLA fine-tuning",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_base_unchanged": frozen_ok,
        "lora_A_updated": a_changed,
        "lora_B_updated": b_changed,
        "steps": steps,
        "interpretation": "B=0 makes grad(A)=0 at the first step; updating B enables grad(A) at step two."
    }
