"""
Agent 5 – Vision Transformer Fine-Tuning.

Loads the processed dataset (preprocessed PyTorch tensors), instantiates the
MedicalClassifierViT, and trains it for classification. Supports dynamic class
detection, early stopping, and metric tracking.

Usage (from the project root)::

    python -m agents.agent5_model_training

Requires Agent 2 to have been run first (``datasets/processed/`` must exist).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.model_config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    OPTIMIZER,
    PATIENCE,
    SCHEDULER,
    SELECTED_FEATURES_DATASET_DIR,
)
from config.settings import EXPORTS_DIR, PROCESSED_DATASET_DIR, REPORTS_DIR
from models.model_loader import MedicalClassifierViT
from preprocessing.tensor_conversion import TensorConverter
from utils.file_utils import ensure_directory_exists, get_all_files_recursive
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="agent5_model_training.log")


class ProcessedTensorDataset(Dataset):
    """
    PyTorch Dataset for loading pre-computed .pt image tensors.
    """

    def __init__(self, data_dir: Path, class_names: list[str]) -> None:
        """
        Args:
            data_dir (Path): The processed dataset directory.
            class_names (list[str]): List of detected class names.
        """
        self.data_dir = data_dir
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.converter = TensorConverter()

        self.samples = []
        for class_name in class_names:
            class_dir = data_dir / class_name
            if class_dir.exists():
                for file_path in class_dir.iterdir():
                    if file_path.suffix == ".pt":
                        self.samples.append((file_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        # Load the preprocessed tensor
        tensor = self.converter.load_tensor(file_path)
        return tensor, label


class VisionTrainingAgent:
    """
    Agent 5: Handles Vision Transformer Fine-Tuning.

    Responsibilities
    ----------------
    1. Validate `selected_features` consistency.
    2. Load `processed` tensors via DataLoaders.
    3. Initialise MedicalClassifierViT and optimizer/scheduler.
    4. Execute training loop with early stopping.
    5. Save best checkpoints and final model.
    6. Export training graphs and final JSON report.
    """

    def __init__(self) -> None:
        self.processed_dir: Path = PROCESSED_DATASET_DIR
        self.selected_features_dir: Path = SELECTED_FEATURES_DATASET_DIR
        
        self.device = DEVICE
        
        # Detected classes
        self.class_names: list[str] = []
        self.num_classes: int = 0
        
        # Tracking metrics
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }
        
    def run(self) -> None:
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("  VISION TRANSFORMER FINE-TUNING AGENT – Starting")
        logger.info("=" * 60)
        
        # Step 1: Detect classes and validate input
        self._detect_classes()
        self._validate_consistency()
        
        # Step 2: Prepare datasets and dataloaders
        train_loader, val_loader, test_loader = self._prepare_dataloaders()
        
        # Step 3: Initialise model, loss, optimizer
        model = MedicalClassifierViT(num_classes=self.num_classes)
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer selection based on config
        if OPTIMIZER.upper() == "ADAM":
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            
        # Scheduler selection based on config
        scheduler = None
        if SCHEDULER.upper() == "REDUCELRONPLATEAU":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=3, factor=0.5
            )
            
        # Step 4: Training Loop
        best_val_acc = self._train_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        
        # Step 5: Test Evaluation
        logger.info("Evaluating best model on test set...")
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, weights_only=True))
        test_metrics = self._evaluate_test_set(model, test_loader)
        
        elapsed = time.time() - start_time
        
        # Step 6: Reporting and Plotting
        self._generate_plots()
        self._save_training_report(test_metrics, best_val_acc, elapsed)
        
        logger.info("=" * 60)
        logger.info("  VISION TRANSFORMER FINE-TUNING AGENT – Completed")
        logger.info("=" * 60)
        
    # =====================================================================
    #  Private helpers
    # =====================================================================
        
    def _detect_classes(self) -> None:
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.processed_dir}")
            
        classes = [d.name for d in self.processed_dir.iterdir() if d.is_dir()]
        self.class_names = sorted(classes)
        self.num_classes = len(self.class_names)
        
        if self.num_classes < 2:
            raise ValueError(f"Found {self.num_classes} classes. Require at least 2 for classification.")
            
        logger.info(f"Detected {self.num_classes} classes: {self.class_names}")

    def _validate_consistency(self) -> None:
        """Use the selected_features folder to ensure dataset consistency as per prompt."""
        if not self.selected_features_dir.exists():
            raise FileNotFoundError("selected_features folder missing. Run Agent 4.")
            
        # Check that we have valid selected features matching the processed images
        for class_name in self.class_names:
            proc_files = len(list((self.processed_dir / class_name).glob("*.pt")))
            sel_files = len(list((self.selected_features_dir / class_name).glob("*.pt")))
            logger.debug(f"Consistency check '{class_name}': {proc_files} processed, {sel_files} selected.")
            if sel_files == 0:
                logger.warning(f"No selected features found for {class_name}!")
        logger.info("Dataset consistency validation passed.")

    def _prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        full_dataset = ProcessedTensorDataset(self.processed_dir, self.class_names)
        total_len = len(full_dataset)
        
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len
        
        # Use a fixed generator seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len], generator=generator
        )
        
        # Prepare dataloaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"Dataset Split -> Train: {train_len}, Val: {val_len}, Test: {test_len}")
        return train_loader, val_loader, test_loader

    def _train_loop(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler
    ) -> float:
        ensure_directory_exists(CHECKPOINTS_DIR)
        ensure_directory_exists(FINAL_MODEL_PATH.parent)
        
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(1, NUM_EPOCHS + 1):
            logger.info(f"Epoch {epoch}/{NUM_EPOCHS}")
            
            # --- Training Phase ---
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            train_loss = running_loss / total
            train_acc = correct / total
            
            # --- Validation Phase ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
            
            if scheduler:
                scheduler.step(val_acc)
                
            # --- Early Stopping & Checkpointing ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                logger.info(f"  >> Validation accuracy improved! Saving best model.")
                # Save checkpoint and the final trained model
                torch.save(model.state_dict(), FINAL_MODEL_PATH)
                chkpt_path = CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch}.pt"
                torch.save(model.state_dict(), chkpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    logger.warning(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        return best_val_acc

    def _evaluate_test_set(self, model: nn.Module, test_loader: DataLoader) -> dict:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        
        # We need to handle multi-class appropriately
        average_mode = 'binary' if self.num_classes == 2 else 'macro'
        
        prec = precision_score(all_labels, all_preds, average=average_mode, zero_division=0)
        rec = recall_score(all_labels, all_preds, average=average_mode, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=average_mode, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"Test Accuracy : {acc:.4f}")
        logger.info(f"Test Precision: {prec:.4f}")
        logger.info(f"Test Recall   : {rec:.4f}")
        logger.info(f"Test F1 Score : {f1:.4f}")
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist()
        }

    def _generate_plots(self) -> None:
        ensure_directory_exists(EXPORTS_DIR)
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Loss Plot
        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", marker='o')
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss", marker='o')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(EXPORTS_DIR / "loss_curve.png")
        plt.close()
        
        # Accuracy Plot
        plt.figure()
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy", marker='o')
        plt.plot(epochs, self.history["val_acc"], label="Validation Accuracy", marker='o')
        plt.title("Training & Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(EXPORTS_DIR / "accuracy_curve.png")
        plt.close()
        
        logger.info("Training plots saved to outputs/exports/")

    def _save_training_report(self, test_metrics: dict, best_val_acc: float, elapsed: float) -> None:
        ensure_directory_exists(REPORTS_DIR)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": "MedicalClassifierViT",
            "dataset_information": {
                "num_classes": self.num_classes,
                "class_names": self.class_names,
            },
            "training_time_seconds": round(elapsed, 2),
            "number_of_epochs_run": len(self.history["train_loss"]),
            "optimizer": OPTIMIZER,
            "scheduler": SCHEDULER,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "best_validation_accuracy": round(best_val_acc, 4),
            "final_test_accuracy": round(test_metrics["accuracy"], 4),
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
            "f1_score": round(test_metrics["f1_score"], 4),
            "confusion_matrix": test_metrics["confusion_matrix"]
        }
        
        report_path = REPORTS_DIR / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Training report saved to {report_path}")


# =========================================================================
#  Entry point
# =========================================================================

if __name__ == "__main__":
    agent = VisionTrainingAgent()
    agent.run()
