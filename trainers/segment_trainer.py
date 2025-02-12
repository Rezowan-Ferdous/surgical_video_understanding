import os
import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import AverageMeter, ScoreMeter, BoundaryScoreMeter
from utils.postprocessing import PostProcessor

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        lambda_bound_loss: float,
        device: str,
        accumulation_steps: int = 4,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion_cls = criterion_cls
        self.criterion_bound = criterion_bound
        self.lambda_bound_loss = lambda_bound_loss
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        mode: str = "ss",
        test_loader: Optional[DataLoader] = None,
    ) -> float:
        self.model.train()
        losses = AverageMeter("Loss", ":.4e")
        total_correct, total_samples = 0, 0

        self.optimizer.zero_grad()
        for i, sample in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            x = sample["feature"].to(self.device)
            t = sample["label"].to(self.device)
            b = sample["boundary"].to(self.device)
            mask = sample["mask"].to(self.device)

            with torch.autocast(device_type=self.device):
                output_cls, output_bound = self.model(x, mask)
                loss = self._compute_loss(output_cls, output_bound, t, b, mask, mode)

            # Gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.update(loss.item() * self.accumulation_steps, x.size(0))
            total_correct, total_samples = self._update_accuracy(
                output_cls, t, mask, total_correct, total_samples, mode
            )

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        logger.info(f"Epoch {epoch + 1}: Loss = {losses.avg}, Accuracy = {accuracy:.4f}")

        if (epoch + 1) % 5 == 0 and test_loader is not None:
            self.test(test_loader, epoch)

        return losses.avg

    def _compute_loss(
        self,
        output_cls: torch.Tensor,
        output_bound: torch.Tensor,
        t: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        loss = 0.0
        if mode == "ms":
            for p in output_cls:
                loss += self.criterion_cls(p, t, x)
        elif isinstance(output_cls, list):
            loss += sum(self.criterion_cls(out, t, x) for out in output_cls) / len(output_cls)
        else:
            loss += self.criterion_cls(output_cls, t, x)

        if isinstance(output_bound, list):
            loss += self.lambda_bound_loss * sum(
                self.criterion_bound(out, b, mask) for out in output_bound
            ) / len(output_bound)
        else:
            loss += self.lambda_bound_loss * self.criterion_bound(output_bound, b, mask)

        return loss

    def _update_accuracy(
        self,
        output_cls: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        total_correct: int,
        total_samples: int,
        mode: str,
    ) -> Tuple[int, int]:
        if mode == "ms":
            _, predicted = torch.max(output_cls[-1], 1)
            total_correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total_samples += torch.sum(mask[:, 0, :]).item()
        return total_correct, total_samples

    def test(self, test_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sample in test_loader:
                x = sample["feature"].to(self.device)
                t = sample["label"].to(self.device)
                mask = sample["mask"].to(self.device)

                output_cls, _ = self.model(x, mask)
                _, predicted = torch.max(output_cls[-1], 1)
                correct += ((predicted == t).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def save_checkpoint(self, epoch: int, best_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["best_loss"]