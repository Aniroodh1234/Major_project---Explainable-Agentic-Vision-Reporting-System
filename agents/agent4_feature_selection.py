"""
Agent 4 – Feature Selection Agent.

Validates deep feature embeddings extracted by Agent 3, ensures they are
not corrupted (no NaNs or Infs) and have correct dimensions, and saves
the verified features to ``datasets/selected_features/``.

Usage (from the project root)::

    python -m agents.agent4_feature_selection

Requires Agent 3 to have been run first (``datasets/features/`` must exist).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.model_config import FEATURES_DATASET_DIR, FEATURE_DIM, SELECTED_FEATURES_DATASET_DIR
from config.settings import EXPECTED_CLASS_FOLDERS, REPORTS_DIR
from preprocessing.tensor_conversion import TensorConverter
from utils.file_utils import clear_directory, ensure_directory_exists, get_all_files_recursive
from utils.logger import setup_logger
from utils.validators import FeatureValidator

logger = setup_logger(__name__, log_file="agent4_feature_selection.log")


class FeatureSelectionAgent:
    """
    Feature Selection Agent for validating feature embeddings.

    Responsibilities
    ----------------
    1. Load extracted feature embeddings from Agent 3.
    2. Validate dimensions, check for corruption (NaNs, Infs).
    3. Ensure labels are correctly mapped.
    4. Remove/filter out invalid records.
    5. Save the valid tensors to ``datasets/selected_features/``.
    6. Generate a comprehensive validation report.
    """

    def __init__(self) -> None:
        """Initialise the agent with paths and the feature validator."""
        self.features_dir: Path = FEATURES_DATASET_DIR
        self.selected_dir: Path = SELECTED_FEATURES_DATASET_DIR
        self.reports_dir: Path = REPORTS_DIR
        self.expected_classes: list[str] = EXPECTED_CLASS_FOLDERS

        self.converter = TensorConverter()
        self.validator = FeatureValidator(expected_dim=FEATURE_DIM)

        # ── Counters & Metadata ─────────────────────────────────────
        self._total_embeddings: int = 0
        self._valid_embeddings: int = 0
        self._removed_embeddings: int = 0
        self._missing_labels: int = 0
        self._inconsistent_dimensions: int = 0
        
        self._metadata_map: list[dict] = []
        self._failed_files: list[dict] = []

    # =====================================================================
    #  Public API
    # =====================================================================

    def run(self) -> dict:
        """
        Execute the feature selection and validation process.

        Returns:
            A dictionary containing the selection report.

        Raises:
            FileNotFoundError: If the feature dataset does not exist.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("  FEATURE SELECTION AGENT – Starting")
        logger.info("=" * 60)
        logger.info(f"Input (features)  : {self.features_dir}")
        logger.info(f"Output (selected) : {self.selected_dir}")
        logger.info(f"Expected dim      : {FEATURE_DIM}")

        # Step 1 — Validate input exists
        self._validate_input()

        # Step 2 — Prepare output directory (fresh start)
        clear_directory(self.selected_dir)

        # Step 3 — Validate and select features
        self._process_and_select_features()

        elapsed = time.time() - start_time

        # Step 4 — Generate and save report
        report = self._generate_report(elapsed)
        self._save_report(report)

        # Step 5 — Log summary
        self._log_summary(elapsed)

        logger.info("=" * 60)
        logger.info("  FEATURE SELECTION AGENT – Completed Successfully")
        logger.info("=" * 60)

        return report

    # =====================================================================
    #  Private helpers
    # =====================================================================

    def _validate_input(self) -> None:
        """
        Verify that the features dataset directory exists.
        """
        if not self.features_dir.exists():
            msg = (
                f"Features dataset not found: {self.features_dir}\n"
                f"Run Agent 3 (feature extraction) first."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Input validation passed.")

    # -----------------------------------------------------------------

    def _process_and_select_features(self) -> None:
        """
        Iterate over the expected class folders, validate each tensor,
        and copy valid tensors to the selected directory.
        """
        for class_name in self.expected_classes:
            class_dir = self.features_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class folder missing: {class_dir}")
                self._missing_labels += 1
                continue
                
            output_class_dir = self.selected_dir / class_name
            ensure_directory_exists(output_class_dir)

            # Target only .pt files
            files = [f for f in get_all_files_recursive(class_dir) if f.suffix == ".pt"]
            logger.info(
                f"Validating features for class '{class_name}': {len(files)} file(s) …"
            )

            class_valid_count = 0
            for file_path in files:
                self._total_embeddings += 1

                try:
                    # 1. Load the feature tensor
                    tensor = self.converter.load_tensor(file_path)

                    # 2. Validate the tensor
                    is_valid, error_msg = self.validator.validate(tensor)

                    if not is_valid:
                        logger.warning(f"Validation failed for {file_path.name}: {error_msg}")
                        self._removed_embeddings += 1
                        
                        if "Invalid shape" in error_msg:
                            self._inconsistent_dimensions += 1
                            
                        self._failed_files.append({
                            "file": str(file_path.relative_to(self.features_dir)),
                            "reason": error_msg
                        })
                        continue

                    # 3. Save valid tensor to selected directory
                    output_name = file_path.name
                    output_path = output_class_dir / output_name
                    self.converter.save_tensor(tensor, output_path)

                    # 4. Maintain clean metadata relationship
                    self._metadata_map.append({
                        "image": output_name,  # The original image identifier
                        "feature_file": str(output_path.relative_to(self.selected_dir)),
                        "label": class_name
                    })

                    self._valid_embeddings += 1
                    class_valid_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    self._removed_embeddings += 1
                    self._failed_files.append({
                        "file": str(file_path.relative_to(self.features_dir)),
                        "reason": f"Exception: {str(e)}"
                    })

            logger.info(
                f"Class '{class_name}' done: {class_valid_count} valid feature(s) selected."
            )

    # -----------------------------------------------------------------

    def _generate_report(self, elapsed_seconds: float) -> dict:
        """Build the feature selection report dictionary."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": str(self.features_dir),
            "output_path": str(self.selected_dir),
            "execution_time_seconds": round(elapsed_seconds, 2),
            "summary": {
                "total_embeddings_evaluated": self._total_embeddings,
                "valid_embeddings_retained": self._valid_embeddings,
                "removed_embeddings": self._removed_embeddings,
                "missing_labels": self._missing_labels,
                "inconsistent_dimensions": self._inconsistent_dimensions,
            },
            "removed_files_details": self._failed_files,
        }

    # -----------------------------------------------------------------

    def _save_report(self, report: dict) -> None:
        """Persist the validation report and full verified metadata as JSON."""
        ensure_directory_exists(self.reports_dir)
        
        # Save the primary summary report
        report_path = self.reports_dir / "feature_selection_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Feature selection report saved: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save feature selection report: {e}")
            raise
            
        # Save the fully clean and mapped metadata explicitly for Agent 5
        metadata_path = self.selected_dir / "selected_features_metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata_map, f, indent=4, ensure_ascii=False)
            logger.info(f"Verified metadata map saved: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save verified metadata map: {e}")

    # -----------------------------------------------------------------

    def _log_summary(self, elapsed_seconds: float) -> None:
        """Print a human-readable summary of the selection results."""
        logger.info("-" * 50)
        logger.info("  FEATURE SELECTION SUMMARY")
        logger.info("-" * 50)
        logger.info(f"  Total embeddings checked: {self._total_embeddings}")
        logger.info(f"  Valid & Selected        : {self._valid_embeddings}")
        logger.info(f"  Removed / Invalid       : {self._removed_embeddings}")
        logger.info(f"  Inconsistent Dimensions : {self._inconsistent_dimensions}")
        logger.info(f"  Execution time          : {elapsed_seconds:.2f}s")
        logger.info("-" * 50)


# =========================================================================
#  Entry point
# =========================================================================

if __name__ == "__main__":
    agent = FeatureSelectionAgent()
    agent.run()
