from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-3)
    return logits / safe_temperature


def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, targets.long()).item())


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = 10,
) -> float:
    flat_probs = probabilities.permute(0, 2, 3, 1).reshape(-1, probabilities.shape[1])
    flat_targets = targets.reshape(-1)
    confidences, predictions = flat_probs.max(dim=1)
    accuracies = predictions.eq(flat_targets)

    boundaries = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probabilities.device)
    ece = torch.tensor(0.0, device=probabilities.device)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = (confidences > start) & (confidences <= end)
        if not torch.any(in_bin):
            continue
        bin_accuracy = accuracies[in_bin].float().mean()
        bin_confidence = confidences[in_bin].mean()
        ece = ece + in_bin.float().mean() * torch.abs(bin_accuracy - bin_confidence)
    return float(ece.item())


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    device = logits.device
    log_temperature = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits / temperature, targets.long())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())


def summarize_temperature_scaling(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> dict[str, float]:
    before_probs = torch.softmax(logits, dim=1)
    after_logits = apply_temperature(logits, temperature)
    after_probs = torch.softmax(after_logits, dim=1)
    return {
        'temperature': float(temperature),
        'nll_before': negative_log_likelihood(logits, targets),
        'nll_after': negative_log_likelihood(after_logits, targets),
        'ece_before': expected_calibration_error(before_probs, targets),
        'ece_after': expected_calibration_error(after_probs, targets),
    }
