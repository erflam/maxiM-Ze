import os
import time
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

from Config import Config
from FileUtils import FileUtils
from FileReader import process_file_checkpoint1
from EICBuilder import process_file_checkpoint2
from Resolving import process_file_checkpoint3, count_peaks_per_file_summary
from PixelMapping import process_file_checkpoint4
from Slicing import process_file_checkpoint5
from Coelution import run_group_coelution
from CoelutionSliced import process_file_coelution_sliced
from Clustering import process_file_cluster_peaks
from Recluster import process_file_recluster
from Visualization import process_visualizations
from ExportExcel import process_export_excel
from LibraryMatching import process_library_match

def init_worker():
    import matplotlib
    matplotlib.use('Agg')
    import os
    os.environ['KALEIDO_DISABLE'] = '1'
    os.environ['PLOTLY_RENDERER'] = 'json'

class Pipeline:
    def __init__(self):
        self.config = Config()
        self.file_paths = FileUtils.get_file_paths()
        Config.initialize_mass_groups(self.file_paths)
        self.file_colors = {
            os.path.splitext(os.path.basename(fp))[0]: FileUtils.random_dark_hex_color()
            for fp in self.file_paths
        }

    def _delete_group_outputs(self, group_name):
        Config.set_mass_group(group_name)
        dirs = Config.setup_directories()

        for key, dir_path in dirs.items():
            if dir_path and os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Deleted [{group_name}] {key}: {dir_path}")

    def clean_run(self):
        """Deletes all previous outputs for ALL groups, then runs fresh."""
        import shutil

        print("\nCLEAN RUN: deleting all previous outputs...\n")

        for group_name in Config.get_group_names_to_run():
            Config.set_mass_group(group_name)
            dirs = Config.setup_directories()

            for dir_path in dirs.values():
                if dir_path and os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted {dir_path}")

        print("\nAll outputs deleted. Starting fresh run...\n")
        self.run()

    def run_group_checkpoint1(self, dirs, group_name):
        files_to_process = [fp for fp in self.file_paths if fp and os.path.exists(fp)]
        if not files_to_process:
            print("No valid files found.")
            return

        start_time = time.time()
        args = [(fp, dirs, group_name) for fp in files_to_process]

        with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
            results = pool.starmap(process_file_checkpoint1, args)

        for r in results:
            print(r)

        elapsed = time.time() - start_time
        print(f'Checkpoint 1 completed in {elapsed:.2f} seconds!')

    def run_group_checkpoint2(self, dirs, group_name):
        files_to_process = [fp for fp in self.file_paths if fp and os.path.exists(fp)]
        if not files_to_process:
            print("No valid files found.")
            return

        start_time = time.time()
        args = [(fp, dirs, self.file_colors, group_name) for fp in files_to_process]

        with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
            results = pool.starmap(process_file_checkpoint2, args)

        for r in results:
            print(r)

        elapsed = time.time() - start_time
        print(f'Checkpoint 2 completed in {elapsed:.2f} seconds!')

    def run_group_checkpoint3(self, dirs, group_name):
        files_to_process = [fp for fp in self.file_paths if fp and os.path.exists(fp)]
        if not files_to_process:
            print("No valid files found.")
            return

        start_time = time.time()
        args = [(fp, dirs, group_name) for fp in files_to_process]

        with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
            results = pool.starmap(process_file_checkpoint3, args)

        for r in results:
            print(r)

        # Summary after resolving
        print(count_peaks_per_file_summary(dirs, group_name))

        elapsed = time.time() - start_time
        print(f'Checkpoint 3 completed in {elapsed:.2f} seconds!')

    def run_group_checkpoint4(self, dirs, group_name):
        png_dir = dirs['png']
        group_tag = str(group_name).replace(" ", "")  # "Group 1" -> "Group1"

        pngs = [
            os.path.join(png_dir, f)
            for f in os.listdir(png_dir)
            if f.startswith("EIC_") and f.endswith(f"_{group_tag}.png")
        ]

        if not pngs:
            print(f"[!] No PNGs found for {group_name} in {png_dir}")
            print("    (Tip) Here are the first 10 files in that folder:")
            try:
                for x in sorted(os.listdir(png_dir))[:10]:
                    print("    -", x)
            except Exception:
                pass
            return

        start_time = time.time()
        args = [(png, dirs, group_name) for png in pngs]

        with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
            results = pool.starmap(process_file_checkpoint4, args)

        for r in results:
            print(r)

        elapsed = time.time() - start_time
        print(f"Checkpoint 4 (Pixel Mapping) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint5(self, dirs, group_name):
        start_time = time.time()
        process_file_checkpoint5(self, dirs, group_name)

        elapsed = time.time() - start_time
        print(f"Checkpoint 5 (Slicing based on Pixel Mapping) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint6(self, dirs, group_name):
        start_time = time.time()
        run_group_coelution(dirs=dirs, group_name=group_name)
        elapsed = time.time() - start_time
        print(f"Checkpoint 6 (Coelution slices added to Directory) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint7(self, dirs, group_name):
        start_time = time.time()
        msg = process_file_coelution_sliced(dirs, group_name)
        print(msg)
        elapsed = time.time() - start_time
        print(f"Checkpoint 7 (Coelution valley reslicing) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint8(self, dirs, group_name):
         start_time = time.time()
         msg = process_file_cluster_peaks(dirs, group_name)
         print(msg)
         elapsed = time.time() - start_time
         print(f"Checkpoint 8 (Peak clustering/alignment) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint9(self, dirs: dict, group_name: str) -> None:
        start_time = time.time()
        msg = process_file_recluster(dirs, group_name)
        print(msg)
        elapsed = time.time() - start_time
        print(f"Checkpoint 9 (Post-clustering RT+mass recluster) completed in {elapsed:.2f} seconds!")

    def run_group_checkpoint10(self, dirs, group_name):
        start_time = time.time()
        self.dirs = dirs  # required for Visualization.py
        msg = process_visualizations(self, group_name)  # <-- FIXED
        print(msg)
        elapsed = time.time() - start_time
        print(f"Checkpoint 10 (Visual QC composites) completed in {elapsed:.2f} seconds!")

    def run_final_checkpoint_excel(self) -> Path:
        start_time = time.time()
        excel_path = process_export_excel(Config)
        print(f"Excel export complete → {excel_path}")
        elapsed = time.time() - start_time
        print(f"Final Checkpoint (Excel export) completed in {elapsed:.2f} seconds!")
        return excel_path

    def run_final_checkpoint_library_match(self, excel_path):
        start_time = time.time()
        msg = process_library_match(Config, excel_path)
        print(msg)
        elapsed = time.time() - start_time
        print(f"Final Checkpoint (Library Match) completed in {elapsed:.2f} seconds!")

    def run(self):
        total_start = time.time()

        for group_name in Config.get_group_names_to_run():
            print(f"Running group: {group_name}")

            Config.set_mass_group(group_name)
            dirs = Config.setup_directories()

            self.run_group_checkpoint1(dirs, group_name)
            self.run_group_checkpoint2(dirs, group_name)
            self.run_group_checkpoint3(dirs, group_name)
            self.run_group_checkpoint4(dirs, group_name)
            self.run_group_checkpoint5(dirs, group_name)
            self.run_group_checkpoint6(dirs, group_name)
            self.run_group_checkpoint7(dirs, group_name)
            self.run_group_checkpoint8(dirs, group_name)
            self.run_group_checkpoint9(dirs, group_name)
            self.run_group_checkpoint10(dirs, group_name)

        excel_path = self.run_final_checkpoint_excel()
        self.run_final_checkpoint_library_match(excel_path)

        total_elapsed = time.time() - total_start

        # Convert to hours, minutes, seconds
        hours = int(total_elapsed // 3600)
        minutes = int((total_elapsed % 3600) // 60)
        seconds = total_elapsed % 60

        print("\n")
        print("🐊" * 33)
        print("=" * 70)
        print(f"ALL GROUPS COMPLETE in {hours:02d}:{minutes:02d}:{seconds:05.2f} (hh:mm:ss)")
        print("=" * 70)
        print("🐊" * 33)
        print("Go Gators! Go Garrett Lab!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    pipeline = Pipeline()
    # Use clean_run() for profiling to delete all previous outputs
    # Use run() for normal execution that skips existing files
    pipeline.clean_run()