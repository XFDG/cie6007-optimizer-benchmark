"""Evaluate a fixed checkpoint; no stochastic layers or optimizer updates."""
import torch
import torch.nn.functional as F

@torch.inference_mode()
def evaluate(model, x, y, batch_size=2048):
    model.eval()
    loss, correct = torch.zeros((), device=x.device), torch.zeros((), device=x.device)
    for start in range(0, len(y), batch_size):
        target = y[start:start+batch_size]
        logits = model(x[start:start+batch_size])
        loss += F.cross_entropy(logits, target, reduction='sum')
        correct += (logits.argmax(1) == target).sum()
    return float(loss / len(y)), float(correct / len(y))
