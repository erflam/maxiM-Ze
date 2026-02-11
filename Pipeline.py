# Pipeline.py
import os
import time
import shutil
from multiprocessing import Pool, cpu_count

from Config import Config
from FileUtils import FileUtils

from FileReader import process_file_checkpoint1
from EICBuilder import process_file_checkpoint2
from Resolving import process_file_checkpoint3, count_peaks_per_file_summary
from PixelMapping import process_file_checkpoint4


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

        for group_name in Config.MASS_GROUPS.keys():
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

    def run(self):
        total_start = time.time()

        for group_name in Config.MASS_GROUPS.keys():
            print("\n" + "=" * 70)
            print(f"Running group: {group_name}")
            print("=" * 70)

            Config.set_mass_group(group_name)
            dirs = Config.setup_directories()

            self.run_group_checkpoint1(dirs, group_name)
            self.run_group_checkpoint2(dirs, group_name)
            self.run_group_checkpoint3(dirs, group_name)
            self.run_group_checkpoint4(dirs, group_name)

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print(f"ALL GROUPS COMPLETE in {total_elapsed:.2f} seconds")
        print("=" * 70)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    pipeline = Pipeline()

    # Use clean_run() for profiling to delete all previous outputs
    # Use run() for normal execution that skips existing files
    pipeline.clean_run()
