"""Fine-tuning engine for the Chinese encoder. torch/transformers are imported lazily."""
import math
import random
import time

from .logic import class_weights, labels_from_scores, macro_f1, select_threshold


def set_seed(seed):
    """Seed python/numpy/torch; GPU kernels stay best-effort deterministic (cuDNN nondeterminism is reported, not hidden)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(config, device):
    """'auto' resolves to bf16 autocast on a capable CUDA device, otherwise plain fp32."""
    if config.get("amp", "auto") != "auto":
        return bool(config.get("amp"))
    if device.type != "cuda":
        return False
    import torch

    return torch.cuda.is_bf16_supported()


def _autocast(device, enabled):
    import torch

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def build_tokenizer(model_name, model_revision):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, revision=model_revision)


def build_model(model_name, model_revision, num_labels):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=model_revision, num_labels=num_labels
    )


def score_texts(model, tokenizer, texts, *, max_length, device, batch_size, amp_enabled):
    """Positive-class (label=1) scores in input order; inference never shuffles."""
    import torch
    from torch.utils.data import DataLoader

    from .data import collate, make_dataset

    model.eval()
    dataset = make_dataset(texts, None, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate(tokenizer))
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, amp_enabled):
                logits = model(**batch).logits
            positive = torch.softmax(logits.float(), dim=-1)[:, 1]
            scores.extend(positive.tolist())
    return scores


def train_encoder(train_texts, train_labels, dev_texts, dev_labels, *, model_name, model_revision, config, device, log):
    """Fine-tune the encoder and select checkpoint + threshold on dev only.

    返回 CPU 上的最优 state_dict、逐epoch dev 指标、dev 选出的阈值与类别权重（仅 train 统计）。
    """
    import torch
    from torch.utils.data import DataLoader

    from .data import collate, make_dataset

    seed = config["seed"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    grad_accum = config["gradient_accumulation_steps"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    warmup_ratio = config.get("warmup_ratio", 0.1)
    weight_decay = config.get("weight_decay", 0.01)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    amp_enabled = resolve_amp(config, device)

    set_seed(seed)
    tokenizer = build_tokenizer(model_name, model_revision)
    model = build_model(model_name, model_revision, config.get("num_labels", 2)).to(device)

    weights = None
    if config.get("class_weight"):
        if config["class_weight"] != "balanced":
            raise ValueError(f"Unsupported class_weight {config['class_weight']!r}; use null or 'balanced'")
        weights = class_weights(train_labels)
        log(f"class_weight=balanced from train only: {weights}")

    from transformers import get_linear_schedule_with_warmup

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for name, p in model.named_parameters() if not any(part in name for part in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for name, p in model.named_parameters() if any(part in name for part in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=learning_rate)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_set = make_dataset(train_texts, train_labels, tokenizer, max_length)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=collate(tokenizer), num_workers=0,
    )
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    loss_weight = torch.tensor(weights, dtype=torch.float32, device=device) if weights else None
    cross_entropy = torch.nn.CrossEntropyLoss(weight=loss_weight)

    best = {"f1": -1.0, "epoch": None, "state": None}
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device, amp_enabled):
                logits = model(**batch).logits
                loss = cross_entropy(logits.float(), batch["labels"])
            (loss / grad_accum).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}/{len(train_loader)} loss {epoch_loss / step:.4f}")

        dev_scores = score_texts(model, tokenizer, dev_texts, max_length=max_length, device=device,
                                 batch_size=config.get("predict_batch_size", 64), amp_enabled=amp_enabled)
        dev_f1 = macro_f1(dev_labels, labels_from_scores(dev_scores, 0.5))
        history.append({"epoch": epoch, "train_loss_mean": epoch_loss / len(train_loader), "dev_macro_f1_at_0.5": dev_f1})
        log(f"epoch {epoch} done: train_loss {epoch_loss / len(train_loader):.4f} dev_macro_f1@0.5 {dev_f1:.4f}")
        if dev_f1 > best["f1"]:
            best = {"f1": dev_f1, "epoch": epoch,
                    "state": {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}}

    if best["state"] is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best["state"])
    dev_scores = score_texts(model, tokenizer, dev_texts, max_length=max_length, device=device,
                             batch_size=config.get("predict_batch_size", 64), amp_enabled=amp_enabled)
    threshold, threshold_f1 = select_threshold(dev_labels, dev_scores)
    log(f"best epoch {best['epoch']} dev_macro_f1@0.5 {best['f1']:.4f}; dev threshold {threshold:.2f} macro_f1 {threshold_f1:.4f}")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "best_epoch": best["epoch"],
        "history": history,
        "dev_macro_f1_at_0.5": best["f1"],
        "threshold": threshold,
        "dev_macro_f1_at_threshold": threshold_f1,
        "class_weights": weights,
        "amp_enabled": amp_enabled,
        "elapsed_seconds": time.time() - started,
    }
