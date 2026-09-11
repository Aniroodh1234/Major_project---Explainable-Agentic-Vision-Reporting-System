from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    CLEANED_DATASET_DIR,
    EXPECTED_CLASS_FOLDERS,
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    PROCESSED_DATASET_DIR,
    REPORTS_DIR,
)
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.tensor_conversion import TensorConverter
from utils.file_utils import (
    clear_directory,
    ensure_directory_exists,
    get_all_files_recursive,
)
from utils.image_utils import is_supported_format
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="agent2_preprocessing.log")


class PreprocessingAgent:
    def __init__(self) -> None:
        """Initialise the agent with paths and pipeline from settings."""
        self.cleaned_dir: Path = CLEANED_DATASET_DIR
        self.processed_dir: Path = PROCESSED_DATASET_DIR
        self.reports_dir: Path = REPORTS_DIR
        self.expected_classes: list[str] = EXPECTED_CLASS_FOLDERS

        # Pipeline in inference mode (no augmentation for batch save)
        self.pipeline: PreprocessingPipeline = PreprocessingPipeline(mode="inference")
        self.converter: TensorConverter = TensorConverter()

        # ── Counters ────────────────────────────────────────────────
        self._total_images: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._class_distribution: dict[str, int] = {}
        self._failed_files: list[str] = []


    def run(self) -> dict:
        """
        Execute the full preprocessing pipeline on the cleaned dataset.

        Returns:
            A dictionary containing the preprocessing report.

        Raises:
            FileNotFoundError: If the cleaned dataset does not exist.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("  PREPROCESSING AGENT – Starting")
        logger.info("=" * 60)
        logger.info(f"Input (cleaned)   : {self.cleaned_dir}")
        logger.info(f"Output (processed): {self.processed_dir}")
        logger.info(
            f"Pipeline config   : size={IMAGE_SIZE}, "
            f"mean={NORMALIZATION_MEAN}, std={NORMALIZATION_STD}"
        )

        # Step 1 — Validate input
        self._validate_input()

        # Step 2 — Prepare output directory (fresh start)
        clear_directory(self.processed_dir)

        # Step 3 — Process all images
        self._process_all_images()

        elapsed = time.time() - start_time

        # Step 4 — Generate and save report
        report = self._generate_report(elapsed)
        self._save_report(report)

        # Step 5 — Log summary
        self._log_summary(elapsed)

        logger.info("=" * 60)
        logger.info("  PREPROCESSING AGENT – Completed Successfully")
        logger.info("=" * 60)

        return report


    def _validate_input(self) -> None:
        """
        Verify that the cleaned dataset directory and class folders exist.

        Raises:
            FileNotFoundError: If any required directory is missing.
        """
        if not self.cleaned_dir.exists():
            msg = (
                f"Cleaned dataset not found: {self.cleaned_dir}\n"
                f"Run Agent 1 (data cleaning) first."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        for class_folder in self.expected_classes:
            class_path = self.cleaned_dir / class_folder
            if not class_path.exists():
                msg = (
                    f"Expected class folder not found: {class_path}\n"
                    f"Ensure Agent 1 has produced the cleaned dataset."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

        logger.info("Input validation passed.")


    def _process_all_images(self) -> None:
        for class_name in self.expected_classes:
            class_dir = self.cleaned_dir / class_name
            output_class_dir = self.processed_dir / class_name
            ensure_directory_exists(output_class_dir)

            files = get_all_files_recursive(class_dir)
            logger.info(
                f"Processing class '{class_name}': {len(files)} file(s) …"
            )

            class_count = 0
            for file_path in files:
                self._total_images += 1

                # Skip non-image files (should not happen in cleaned/)
                if not is_supported_format(file_path):
                    logger.debug(f"Skipping non-image: {file_path.name}")
                    continue

                # Run the pipeline
                tensor = self.pipeline.process_from_path(file_path)
                if tensor is None:
                    self._failed += 1
                    self._failed_files.append(
                        str(file_path.relative_to(self.cleaned_dir))
                    )
                    continue

                # Save as .pt file (same base name, new extension)
                output_name = file_path.stem + ".pt"
                output_path = output_class_dir / output_name
                self.converter.save_tensor(tensor, output_path)

                self._successful += 1
                class_count += 1

            self._class_distribution[class_name] = class_count
            logger.info(
                f"Class '{class_name}' done: {class_count} tensor(s) saved."
            )

    def _generate_report(self, elapsed_seconds: float) -> dict:
        """Build the preprocessing report dictionary."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": str(self.cleaned_dir),
            "output_path": str(self.processed_dir),
            "execution_time_seconds": round(elapsed_seconds, 2),
            "pipeline_config": {
                "image_size": IMAGE_SIZE,
                "normalization_mean": NORMALIZATION_MEAN,
                "normalization_std": NORMALIZATION_STD,
                "mode": "inference",
            },
            "summary": {
                "total_images": self._total_images,
                "successfully_processed": self._successful,
                "failed": self._failed,
            },
            "class_distribution": self._class_distribution,
            "failed_files": self._failed_files,
        }

    def _save_report(self, report: dict) -> None:
        """Persist the preprocessing report as JSON."""
        ensure_directory_exists(self.reports_dir)
        report_path = self.reports_dir / "preprocessing_report.json"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Preprocessing report saved: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessing report: {e}")
            raise

    def _log_summary(self, elapsed_seconds: float) -> None:
        """Print a human-readable summary of the preprocessing results."""
        logger.info("-" * 50)
        logger.info("  PREPROCESSING SUMMARY")
        logger.info("-" * 50)
        logger.info(f"  Total images         : {self._total_images}")
        logger.info(f"  Successfully processed: {self._successful}")
        logger.info(f"  Failed               : {self._failed}")
        logger.info(f"  Class distribution   : {self._class_distribution}")
        logger.info(f"  Output format        : PyTorch tensor (.pt)")
        logger.info(f"  Tensor shape         : (3, {IMAGE_SIZE}, {IMAGE_SIZE})")
        logger.info(f"  Execution time       : {elapsed_seconds:.2f}s")
        logger.info("-" * 50)


if __name__ == "__main__":
    agent = PreprocessingAgent()
    agent.run()
