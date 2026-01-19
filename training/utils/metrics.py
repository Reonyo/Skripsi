import torch


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def _f1_binary(preds: torch.Tensor, labels: torch.Tensor) -> float:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mcc_binary(preds: torch.Tensor, labels: torch.Tensor) -> float:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    numerator = tp * tn - fp * fn
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return _safe_div(numerator, denom)


def _pearson(preds: torch.Tensor, labels: torch.Tensor) -> float:
    x = preds - preds.mean()
    y = labels - labels.mean()
    num = (x * y).sum()
    den = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt())
    return _safe_div(num.item(), den.item())


def _spearman(preds: torch.Tensor, labels: torch.Tensor) -> float:
    def rank(t: torch.Tensor) -> torch.Tensor:
        sorted_idx = torch.argsort(t)
        ranks = torch.zeros_like(sorted_idx, dtype=torch.float)
        ranks[sorted_idx] = torch.arange(len(t), dtype=torch.float, device=t.device)
        return ranks

    rx = rank(preds)
    ry = rank(labels)
    return _pearson(rx, ry)


def compute_metrics(task_name: str, num_labels: int, logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute GLUE-style metrics for validation."""
    metrics = {
        "val_accuracy": 0.0,
        "val_f1": 0.0,
        "val_mcc": 0.0,
        "val_pearson": 0.0,
        "val_spearman": 0.0,
    }

    if num_labels == 1:
        preds = logits.squeeze(-1)
        metrics["val_pearson"] = _pearson(preds, labels)
        metrics["val_spearman"] = _spearman(preds, labels)
        return metrics

    pred_labels = torch.argmax(logits, dim=-1)
    metrics["val_accuracy"] = _accuracy(pred_labels, labels)

    if task_name.upper() == "COLA":
        metrics["val_mcc"] = _mcc_binary(pred_labels, labels)
    elif task_name.upper() in {"QQP", "MRPC"}:
        metrics["val_f1"] = _f1_binary(pred_labels, labels)

    return metrics
