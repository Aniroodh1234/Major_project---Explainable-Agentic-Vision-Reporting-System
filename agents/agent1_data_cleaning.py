"""
Agent 1 – Data Cleaning Agent.

Scans the raw medical image dataset, removes corrupted / unsupported /
duplicate files, validates folder structure, and writes a clean copy
to ``datasets/cleaned/`` while preserving the original class hierarchy.

A JSON cleaning report is saved to ``outputs/reports/cleaning_report.json``.

Usage (from the project root)::

    python -m agents.agent1_data_cleaning

The raw dataset is **never** modified.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that config / utils are importable
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    ALL_IMAGE_EXTENSIONS,
    CLEANED_DATASET_DIR,
    EXPECTED_CLASS_FOLDERS,
    RAW_DATASET_DIR,
    REPORTS_DIR,
    SUPPORTED_IMAGE_FORMATS,
)
from utils.file_utils import (
    clear_directory,
    copy_file,
    ensure_directory_exists,
    get_all_files_recursive,
    get_file_extension,
    is_hidden_path,
)
from utils.image_utils import compute_image_hash, is_supported_format, is_valid_image
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="agent1_data_cleaning.log")


class DataCleaningAgent:
    """
    Reusable Data Cleaning Agent for the medical image analysis pipeline.

    Responsibilities
    ----------------
    1. Validate the expected folder structure.
    2. Scan the raw dataset recursively.
    3. Ignore hidden files.
    4. Detect and flag unsupported image formats.
    5. Detect and flag corrupted images (cannot be opened by OpenCV).
    6. Detect and remove duplicate images via perceptual hashing.
    7. Copy valid, unique images to ``datasets/cleaned/``.
    8. Generate a JSON cleaning report.

    The raw dataset is **never** overwritten or modified.
    """

    def __init__(self) -> None:
        """Initialise the agent with paths and counters from settings."""
        # Paths (from config)
        self.raw_dir: Path = RAW_DATASET_DIR
        self.cleaned_dir: Path = CLEANED_DATASET_DIR
        self.reports_dir: Path = REPORTS_DIR
        self.expected_classes: list[str] = EXPECTED_CLASS_FOLDERS

        # ── Counters ────────────────────────────────────────────────
        self._total_files_scanned: int = 0
        self._total_images: int = 0
        self._valid_images: int = 0
        self._corrupted_images: int = 0
        self._duplicate_images_removed: int = 0
        self._unsupported_files: int = 0
        self._hidden_files_ignored: int = 0
        self._non_image_files_ignored: int = 0
        self._final_dataset_size: int = 0
        self._class_distribution: dict[str, int] = {}

        # ── Detailed lists for the report ───────────────────────────
        self._corrupted_files_list: list[str] = []
        self._unsupported_files_list: list[str] = []
        self._duplicate_files_removed_list: list[str] = []

    # =====================================================================
    #  Public API
    # =====================================================================

    def run(self) -> dict:
        """
        Execute the full data-cleaning pipeline.

        Returns:
            A dictionary containing the cleaning report (also saved as JSON).

        Raises:
            FileNotFoundError: If the expected folder structure is missing.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("  DATA CLEANING AGENT – Starting")
        logger.info("=" * 60)
        logger.info(f"Raw dataset path  : {self.raw_dir}")
        logger.info(f"Cleaned output    : {self.cleaned_dir}")

        # Step 1 — Validate folder structure
        self._validate_folder_structure()

        # Step 2 — Prepare output directory (fresh start)
        clear_directory(self.cleaned_dir)

        # Step 3 — Scan, filter, and validate all files
        valid_images_by_class = self._scan_and_filter()

        # Step 4 — Remove duplicate images (global dedup via hashing)
        unique_images_by_class = self._remove_duplicates(valid_images_by_class)

        # Step 5 — Copy clean images to cleaned/ directory
        self._save_cleaned_images(unique_images_by_class)

        # Step 6 — Compute final stats
        self._final_dataset_size = sum(
            len(imgs) for imgs in unique_images_by_class.values()
        )
        for class_name, imgs in unique_images_by_class.items():
            self._class_distribution[class_name] = len(imgs)

        elapsed = time.time() - start_time

        # Step 7 — Generate and persist the report
        report = self._generate_report(elapsed)
        self._save_report(report)

        # Step 8 — Log summary
        self._log_summary(elapsed)

        logger.info("=" * 60)
        logger.info("  DATA CLEANING AGENT – Completed Successfully")
        logger.info("=" * 60)

        return report

    # =====================================================================
    #  Private helpers
    # =====================================================================

    def _validate_folder_structure(self) -> None:
        """
        Verify that the raw dataset directory and every expected class
        folder exist.

        Raises:
            FileNotFoundError: With a descriptive message when any
                required directory is absent.
        """
        if not self.raw_dir.exists():
            msg = (
                f"Raw dataset directory not found: {self.raw_dir}\n"
                f"Expected structure:\n"
                f"  datasets/\n"
                f"      raw/\n"
                f"          with_cancer/\n"
                f"          without_cancer/"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        for class_folder in self.expected_classes:
            class_path = self.raw_dir / class_folder
            if not class_path.exists():
                msg = (
                    f"Expected class folder not found: {class_path}\n"
                    f"The raw dataset must contain a '{class_folder}' folder."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

        logger.info("Folder structure validated successfully.")

    # -----------------------------------------------------------------

    def _scan_and_filter(self) -> dict[str, list[Path]]:
        """
        Walk every class folder inside ``raw/``, categorise each file,
        and return valid image paths grouped by class.

        Categories
        ----------
        * **hidden**       – file (or a parent dir) name starts with ``.``
        * **unsupported**  – known image extension but not in supported list
        * **non-image**    – extension is not a known image extension at all
        * **corrupted**    – supported extension but OpenCV cannot read it
        * **valid**        – all checks passed

        Returns:
            ``{class_name: [Path, …]}`` of valid, uncorrupted images.
        """
        valid_images: dict[str, list[Path]] = {
            cls: [] for cls in self.expected_classes
        }

        for class_name in self.expected_classes:
            class_dir = self.raw_dir / class_name
            files = get_all_files_recursive(class_dir)
            logger.info(
                f"Scanning class '{class_name}': {len(files)} file(s) found."
            )

            for file_path in files:
                self._total_files_scanned += 1

                # 1. Hidden file / hidden parent directory?
                if is_hidden_path(file_path, self.raw_dir):
                    self._hidden_files_ignored += 1
                    logger.debug(f"Hidden file ignored: {file_path.name}")
                    continue

                ext = get_file_extension(file_path)

                # 2. Is it a supported image format?
                if not is_supported_format(file_path):
                    # Distinguish "unsupported image" vs "non-image"
                    if ext in ALL_IMAGE_EXTENSIONS:
                        self._unsupported_files += 1
                        self._unsupported_files_list.append(
                            str(file_path.relative_to(self.raw_dir))
                        )
                        logger.debug(
                            f"Unsupported image format: {file_path.name}"
                        )
                    else:
                        self._non_image_files_ignored += 1
                        logger.debug(f"Non-image file ignored: {file_path.name}")
                    continue

                # From here on, the file has a supported extension.
                self._total_images += 1

                # 3. Can OpenCV actually read it?
                if not is_valid_image(file_path):
                    self._corrupted_images += 1
                    self._corrupted_files_list.append(
                        str(file_path.relative_to(self.raw_dir))
                    )
                    logger.warning(f"Corrupted image: {file_path.name}")
                    continue

                # Image is valid.
                self._valid_images += 1
                valid_images[class_name].append(file_path)

        logger.info(
            f"Scan complete — "
            f"total files: {self._total_files_scanned}, "
            f"supported images: {self._total_images}, "
            f"valid: {self._valid_images}, "
            f"corrupted: {self._corrupted_images}, "
            f"unsupported: {self._unsupported_files}, "
            f"hidden: {self._hidden_files_ignored}, "
            f"non-image: {self._non_image_files_ignored}"
        )
        return valid_images

    # -----------------------------------------------------------------

    def _remove_duplicates(
        self,
        images_by_class: dict[str, list[Path]],
    ) -> dict[str, list[Path]]:
        """
        Remove duplicate images using perceptual hashing (dHash).

        Deduplication is performed **globally** across all classes so
        that an identical image accidentally placed in both
        ``with_cancer/`` and ``without_cancer/`` is caught.
        The **first** occurrence (in alphabetical scan order) is kept.

        Args:
            images_by_class: ``{class_name: [Path, …]}`` of valid images.

        Returns:
            A new dictionary with the same structure but duplicates removed.
        """
        logger.info("Starting duplicate detection via image hashing …")

        seen_hashes: dict[str, Path] = {}
        unique_images: dict[str, list[Path]] = {
            cls: [] for cls in self.expected_classes
        }

        for class_name in self.expected_classes:
            for file_path in images_by_class[class_name]:
                img_hash = compute_image_hash(file_path)

                if img_hash is None:
                    # Could not hash – treat as valid to avoid data loss.
                    logger.warning(
                        f"Could not hash {file_path.name}; keeping it."
                    )
                    unique_images[class_name].append(file_path)
                    continue

                if img_hash in seen_hashes:
                    # Duplicate found.
                    self._duplicate_images_removed += 1
                    self._duplicate_files_removed_list.append(
                        str(file_path.relative_to(self.raw_dir))
                    )
                    logger.debug(
                        f"Duplicate removed: {file_path.name} "
                        f"(matches {seen_hashes[img_hash].name})"
                    )
                else:
                    seen_hashes[img_hash] = file_path
                    unique_images[class_name].append(file_path)

        total_unique = sum(len(v) for v in unique_images.values())
        logger.info(
            f"Duplicate detection complete — "
            f"{self._duplicate_images_removed} duplicate(s) removed, "
            f"{total_unique} unique image(s) remaining."
        )
        return unique_images

    # -----------------------------------------------------------------

    def _save_cleaned_images(
        self,
        images_by_class: dict[str, list[Path]],
    ) -> None:
        """
        Copy unique, valid images to ``datasets/cleaned/`` preserving
        the original class-folder hierarchy.

        Args:
            images_by_class: ``{class_name: [Path, …]}`` of unique images.
        """
        logger.info(f"Saving cleaned images to: {self.cleaned_dir}")
        copy_success = 0
        copy_fail = 0

        for class_name, file_paths in images_by_class.items():
            dest_class_dir = self.cleaned_dir / class_name
            ensure_directory_exists(dest_class_dir)

            for src in file_paths:
                dst = dest_class_dir / src.name
                if copy_file(src, dst):
                    copy_success += 1
                else:
                    copy_fail += 1

        logger.info(
            f"Saved {copy_success} image(s) to cleaned directory "
            f"({copy_fail} copy failure(s))."
        )

    # -----------------------------------------------------------------

    def _generate_report(self, elapsed_seconds: float) -> dict:
        """
        Build the cleaning report dictionary.

        Args:
            elapsed_seconds: Total pipeline execution time in seconds.

        Returns:
            Report dictionary ready for JSON serialisation.
        """
        report: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_dataset_path": str(self.raw_dir),
            "cleaned_dataset_path": str(self.cleaned_dir),
            "execution_time_seconds": round(elapsed_seconds, 2),
            "summary": {
                "total_files_scanned": self._total_files_scanned,
                "total_images": self._total_images,
                "valid_images": self._valid_images,
                "corrupted_images": self._corrupted_images,
                "duplicate_images_removed": self._duplicate_images_removed,
                "unsupported_files": self._unsupported_files,
                "hidden_files_ignored": self._hidden_files_ignored,
                "non_image_files_ignored": self._non_image_files_ignored,
                "final_dataset_size": self._final_dataset_size,
            },
            "class_distribution": self._class_distribution,
            "details": {
                "corrupted_files": self._corrupted_files_list,
                "unsupported_files": self._unsupported_files_list,
                "duplicate_files_removed": self._duplicate_files_removed_list,
            },
        }
        return report

    # -----------------------------------------------------------------

    def _save_report(self, report: dict) -> None:
        """
        Persist the cleaning report as JSON.

        The report is saved to ``outputs/reports/cleaning_report.json``.

        Args:
            report: The report dictionary to serialise.
        """
        ensure_directory_exists(self.reports_dir)
        report_path = self.reports_dir / "cleaning_report.json"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Cleaning report saved: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save cleaning report: {e}")
            raise

    # -----------------------------------------------------------------

    def _log_summary(self, elapsed_seconds: float) -> None:
        """
        Print a human-readable summary of the cleaning results.

        Args:
            elapsed_seconds: Total execution time.
        """
        logger.info("-" * 50)
        logger.info("  CLEANING SUMMARY")
        logger.info("-" * 50)
        logger.info(f"  Total files scanned      : {self._total_files_scanned}")
        logger.info(f"  Total images (supported) : {self._total_images}")
        logger.info(f"  Valid images             : {self._valid_images}")
        logger.info(f"  Corrupted images         : {self._corrupted_images}")
        logger.info(f"  Duplicates removed       : {self._duplicate_images_removed}")
        logger.info(f"  Unsupported files        : {self._unsupported_files}")
        logger.info(f"  Hidden files ignored     : {self._hidden_files_ignored}")
        logger.info(f"  Non-image files ignored  : {self._non_image_files_ignored}")
        logger.info(f"  Final dataset size       : {self._final_dataset_size}")
        logger.info(f"  Class distribution       : {self._class_distribution}")
        logger.info(f"  Execution time           : {elapsed_seconds:.2f}s")
        logger.info("-" * 50)


# =========================================================================
#  Entry point
# =========================================================================

if __name__ == "__main__":
    agent = DataCleaningAgent()
    agent.run()
