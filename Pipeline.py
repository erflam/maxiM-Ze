"""
Pipeline.py
Main orchestration script for the mass spectrometry analysis pipeline.

Pipeline checkpoints:
1. FileReader: Process mzXML files and create filtered CSVs for each mass group
2. EICBuilder: Generate EIC PNG images with peak detection for each mass group
3. (Future checkpoints can be added here)
"""

import time
import shutil
from pathlib import Path
from typing import Optional, List

from Config import Config
from FileUtils import FileUtils
import FileReader
import EICBuilder


class Pipeline:
    """Main pipeline orchestrator for MS analysis"""

    def __init__(self, clean_run: bool = False):
        """
        Initialize the pipeline

        Args:
            clean_run: If True, delete all existing output before starting
        """
        self.start_time = None
        self.checkpoint_times = {}
        self.clean_run = clean_run

        if self.clean_run:
            self.clean_output_directory()

    def clean_output_directory(self):
        """Delete all existing output to start fresh"""
        output_root = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER

        if output_root.exists():
            print("\n" + "=" * 80)
            print("CLEAN RUN: Removing existing output")
            print("=" * 80)
            print(f"Deleting: {output_root}")

            try:
                shutil.rmtree(output_root)
                print("[✔] Output directory cleaned successfully")
            except Exception as e:
                print(f"[!] Error cleaning output directory: {e}")
                raise
        else:
            print("\n[i] No existing output found (clean start)")

    def format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable way

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f} hours ({minutes:.2f} minutes, {seconds:.2f} seconds)"

    def run_checkpoint_1_file_reading(self, noise_level: float = 5000.0, n_processes: Optional[int] = None):
        """
        Checkpoint 1: Read mzXML files and create filtered CSVs for all groups

        Args:
            noise_level: Minimum intensity threshold for peak detection
            n_processes: Number of parallel processes (default: CPU count - 1)
        """
        print("\n" + "=" * 80)
        print("CHECKPOINT 1: FILE READING & PEAK DETECTION")
        print("=" * 80)

        checkpoint_start = time.time()

        # Run FileReader to process all files and create filtered CSVs for all groups
        FileReader.process_all_groups(
            noise_level=noise_level,
            n_processes=n_processes
        )

        checkpoint_elapsed = time.time() - checkpoint_start
        self.checkpoint_times['Checkpoint 1: File Reading'] = checkpoint_elapsed

        print(f"\n[✔] Checkpoint 1 completed in {self.format_time(checkpoint_elapsed)}")

    def run_checkpoint_2_eic_building(self, n_processes: Optional[int] = None):
        """
        Checkpoint 2: Build EIC PNG images with peak detection for all groups

        Args:
            n_processes: Number of parallel processes (default: CPU count - 1)
        """
        print("\n" + "=" * 80)
        print("CHECKPOINT 2: EIC IMAGE GENERATION")
        print("=" * 80)

        checkpoint_start = time.time()

        # Run EICBuilder to generate PNG images for all groups
        EICBuilder.build_eics_for_all_groups(n_processes=n_processes)

        checkpoint_elapsed = time.time() - checkpoint_start
        self.checkpoint_times['Checkpoint 2: EIC Building'] = checkpoint_elapsed

        print(f"\n[✔] Checkpoint 2 completed in {self.format_time(checkpoint_elapsed)}")

    def run_full_pipeline(
            self,
            noise_level: float = 5000.0,
            n_processes: Optional[int] = None,
            checkpoints: Optional[List[int]] = None
    ):
        """
        Run the complete analysis pipeline for all groups

        Args:
            noise_level: Minimum intensity threshold for peak detection
            n_processes: Number of parallel processes (default: CPU count - 1)
            checkpoints: List of checkpoint numbers to run (default: all checkpoints [1, 2])
        """
        self.start_time = time.time()

        if checkpoints is None:
            checkpoints = [1, 2]  # Run all checkpoints by default

        print("\n" + "=" * 80)
        print("MASS SPECTROMETRY ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Clean Run: {'YES (starting fresh)' if self.clean_run else 'NO (incremental)'}")
        print(f"Analysis Folder: {Config.ANALYSIS_FOLDER}")
        print(f"Base Directory: {Config.BASE_DIR}")
        print(f"Input Directory: {Config.BASE_DIR / Config.INPUT_SUBDIR}")
        print(f"Output Root: {Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER}")
        print(f"Number of Groups: {len(Config.MASS_GROUPS)}")
        print(f"Groups: {', '.join(Config.MASS_GROUPS.keys())}")
        print(f"Checkpoints to run: {checkpoints}")
        print("=" * 80)

        # Checkpoint 1: File Reading
        if 1 in checkpoints:
            self.run_checkpoint_1_file_reading(noise_level=noise_level, n_processes=n_processes)

        # Checkpoint 2: EIC Building
        if 2 in checkpoints:
            self.run_checkpoint_2_eic_building(n_processes=n_processes)

        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print pipeline execution summary"""
        total_elapsed = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print()

        # Print individual checkpoint times
        for checkpoint_name, checkpoint_time in self.checkpoint_times.items():
            print(f"{checkpoint_name}:")
            print(f"  Time: {self.format_time(checkpoint_time)}")
            print()

        # Print total time
        print("-" * 80)
        print(f"Total Pipeline Time: {self.format_time(total_elapsed)}")
        print("-" * 80)

        # Calculate percentage breakdown if multiple checkpoints ran
        if len(self.checkpoint_times) > 1:
            print("\nTime Breakdown:")
            for checkpoint_name, checkpoint_time in self.checkpoint_times.items():
                percentage = (checkpoint_time / total_elapsed) * 100
                print(f"  {checkpoint_name}: {percentage:.1f}%")

        print("=" * 80)

        # Print output locations
        print("\nOutput Locations:")
        print(f"  Raw Peak Data: {Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / 'Mass Detection'}")

        for group_name in Config.MASS_GROUPS.keys():
            group_dir = Config.BASE_DIR / Config.OUTPUT_ROOT / Config.ANALYSIS_FOLDER / group_name
            print(f"\n  {group_name}:")
            print(f"    Filtered CSVs: {group_dir / 'EIC CSVs'}")
            print(f"    EIC Images: {group_dir / 'EIC PNGs'}")

        print("\n" + "=" * 80)


def main():
    """Main entry point for the pipeline"""

    # ========================================
    # OPTION 1: Full clean run (start fresh) - ALL GROUPS
    # ========================================
    pipeline = Pipeline(clean_run=True)
    pipeline.run_full_pipeline(
        noise_level=5000.0,
        n_processes=None,  # Auto-detect CPU cores
        checkpoints=[1, 2]  # Run all checkpoints
    )

    # ========================================
    # OPTION 2: Incremental run (use existing data where possible) - ALL GROUPS
    # ========================================
    # pipeline = Pipeline(clean_run=False)
    # pipeline.run_full_pipeline(
    #     noise_level=5000.0,
    #     n_processes=None,
    #     checkpoints=[1, 2]
    # )

    # ========================================
    # OPTION 3: Run only specific checkpoints - ALL GROUPS
    # ========================================
    # pipeline = Pipeline(clean_run=False)
    # pipeline.run_full_pipeline(
    #     noise_level=5000.0,
    #     n_processes=None,
    #     checkpoints=[2]  # Only run Checkpoint 2 (EIC building)
    # )


if __name__ == "__main__":
    main()