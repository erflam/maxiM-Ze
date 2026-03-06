from pathlib import Path
import csv
import json

class Config:
    BASE_DIR = Path.home() / "Desktop/maxiMiZe Tests"
    INPUT_SUBDIR = Path("maxiMiZe Files")
    OUTPUT_ROOT = Path("maxiMiZe Checkpoints")
    ANALYSIS_FOLDER = "maxiMZe Group 1-30 0306 Test 1"

    USE_DYNAMIC_MASS_GROUPS = True         # MUST be True (fallback groups removed)
    MAX_GROUPS_TO_RUN = 30                # None = all, N = first N groups
    REBUILD_MASS_GROUPS = True            # True = ignore cache and recompute
    GROUPING_VERBOSE = False               # True = print per-file processing during grouping

    GROUP_NOISE_LEVEL = 5000.0
    GROUP_MZ_TOLERANCE = 0.0005
    GROUP_MIN_CONSEC_SCANS = 7
    GROUP_MIN_SAMPLE_PRESENCE = 1
    GROUP_MIN_GROUP_SIZE = 3
    GROUP_MAX_GROUP_SIZE = 5

    MASS_GROUPS_CACHE_NAME = "MassGroups_Cache.json"
    MASS_GROUPS_EXPORT_NAME = "MassGroups_Formatted.csv"
    MASS_GROUPS: dict[str, list[float]] = {}   # populated by initialize_mass_groups() or cache
    MASS_LIST: list[float] = []               # active group's masses
    CURRENT_GROUP: str | None = None
    MASS_TOLERANCE = 0.0005
    MAX_PEAK_DURATION = 1.5

    # Run specific masses only
    RUN_ONLY_MASSES = None  # Set to None to run normal dynamic groups

    # Library Matching
    LIB_FILE = r"/Users/elizabethflammer/Desktop/Research/MZMine/POS OE Library Metformin Baseline.csv"
    LIBRARY_MATCH_MZ_TOL = 0.0005
    LIBRARY_MATCH_RT_TOL = 0.1

    @classmethod
    def _analysis_output_root(cls) -> Path:
        # User requested:
        # output_root = Path(Config.BASE_DIR) / Path(Config.OUTPUT_ROOT) / Config.ANALYSIS_FOLDER
        return Path(cls.BASE_DIR) / Path(cls.OUTPUT_ROOT) / cls.ANALYSIS_FOLDER

    @classmethod
    def _mass_groups_cache_path(cls) -> Path:
        out_dir = cls._analysis_output_root()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / cls.MASS_GROUPS_CACHE_NAME

    @staticmethod
    def _group_sort_key(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 10**9

    @staticmethod
    def _display_group_label(name: str) -> str:
        digits = "".join(ch for ch in name if ch.isdigit())
        return f"'Group {digits}':" if digits else f"'{name}':"

    @classmethod
    def save_mass_groups_cache(cls) -> None:
        p = cls._mass_groups_cache_path()
        p.write_text(json.dumps(cls.MASS_GROUPS, indent=2), encoding="utf-8")

    @classmethod
    def load_mass_groups_cache(cls) -> bool:
        p = cls._mass_groups_cache_path()
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        cls.MASS_GROUPS = {k: [float(x) for x in v] for k, v in data.items()}
        return True

    @classmethod
    def ensure_mass_groups_loaded(cls) -> None:
        """
        Used by multiprocessing workers.
        Workers won't inherit runtime-modified class variables from the parent
        (spawn), so they must load MASS_GROUPS from disk.
        """
        if cls.MASS_GROUPS:
            return
        if not cls.load_mass_groups_cache():
            raise RuntimeError(
                "MASS_GROUPS not initialized in this process and cache not found.\n"
                f"Expected cache at: {cls._mass_groups_cache_path()}\n"
                "Fix: make sure the main process calls Config.initialize_mass_groups(...) "
                "before starting multiprocessing."
            )

    @classmethod
    def initialize_mass_groups(cls, _pipeline_file_paths_ignored: list[str] | None = None) -> None:
        """
        Builds MASS_GROUPS using what is defined in MassGrouping.py:
        - select_files() (study design + manifest) -> typically 6 files
        - build_mass_groups_from_files() -> groups

        Also writes:
        - MassGroups_Cache.json  (for multiprocessing workers)
        - MassGroups_Formatted.csv
        """

        if cls.RUN_ONLY_MASSES is not None:
            masses = [round(float(m), 4) for m in cls.RUN_ONLY_MASSES]

            cls.MASS_GROUPS = {"Group1": masses}
            cls.CURRENT_GROUP = "Group1"
            cls.MASS_LIST = masses

            cls.save_mass_groups_cache()
            cls.save_mass_groups_formatted_csv()
            return

        if not cls.USE_DYNAMIC_MASS_GROUPS:
            raise ValueError("USE_DYNAMIC_MASS_GROUPS=False but fallback MASS_GROUPS were removed.")

        # If we already have groups in memory and we're not rebuilding, keep them.
        if cls.MASS_GROUPS and not cls.REBUILD_MASS_GROUPS:
            cls.save_mass_groups_formatted_csv()
            return

        # Try cache first (fast)
        if not cls.REBUILD_MASS_GROUPS and cls.load_mass_groups_cache():
            # set default current group
            first_group = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)[0]
            cls.set_mass_group(first_group)
            cls.save_mass_groups_formatted_csv()
            return

        # Build fresh (slow)
        from MassGrouping import select_files, build_mass_groups_from_files

        grouping_files = select_files()

        cls.MASS_GROUPS = build_mass_groups_from_files(
            grouping_files,
            noise_level=cls.GROUP_NOISE_LEVEL,
            mz_tolerance=cls.GROUP_MZ_TOLERANCE,
            min_consec_scans=cls.GROUP_MIN_CONSEC_SCANS,
            min_sample_presence=cls.GROUP_MIN_SAMPLE_PRESENCE,
            min_group_size=cls.GROUP_MIN_GROUP_SIZE,
            max_group_size=cls.GROUP_MAX_GROUP_SIZE,
            # If you add verbose support to MassGrouping.build_mass_groups_from_files, it will use this:
            verbose=getattr(cls, "GROUPING_VERBOSE", False),
        )

        if not cls.MASS_GROUPS:
            raise ValueError("Dynamic mass grouping produced 0 groups.")

        # Set default current group
        first_group = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)[0]
        cls.set_mass_group(first_group)

        # Persist for multiprocessing workers + export CSV
        cls.save_mass_groups_cache()
        cls.save_mass_groups_formatted_csv()

    @classmethod
    def get_group_names_to_run(cls) -> list[str]:
        cls.ensure_mass_groups_loaded()
        names = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)
        if cls.MAX_GROUPS_TO_RUN is None:
            return names
        return names[: int(cls.MAX_GROUPS_TO_RUN)]

    @classmethod
    def set_mass_group(cls, group_name: str) -> None:
        # Multiprocessing-safe: if a worker calls this first, it will load cache.
        cls.ensure_mass_groups_loaded()

        if group_name not in cls.MASS_GROUPS:
            available = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)
            raise ValueError(
                f"Invalid group name: {group_name}. "
                f"First available groups: {available[:10]}"
            )
        cls.CURRENT_GROUP = group_name
        cls.MASS_LIST = cls.MASS_GROUPS[group_name]

    @classmethod
    def save_mass_groups_formatted_csv(cls) -> Path:
        """
        Saves ALL groups (not limited by MAX_GROUPS_TO_RUN) to:
          BASE_DIR / OUTPUT_ROOT / ANALYSIS_FOLDER / MassGroups_Formatted.csv
        """
        cls.ensure_mass_groups_loaded()

        out_dir = cls._analysis_output_root()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cls.MASS_GROUPS_EXPORT_NAME

        group_names = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "Size", "Masses"])
            for name in group_names:
                masses = cls.MASS_GROUPS.get(name, [])
                masses_str = "[" + ", ".join(f"{float(m):.4f}" for m in masses) + "]"
                writer.writerow([cls._display_group_label(name), len(masses), masses_str])

        return out_path

    @classmethod
    def setup_directories(cls):
        if cls.CURRENT_GROUP is None:
            raise RuntimeError(
                "CURRENT_GROUP is None. "
                "Call Config.initialize_mass_groups(...) and Config.set_mass_group(...) first."
            )

        dirs = {
            'png': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "EIC PNGs",
            'csv': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "EIC CSVs",
            'slice': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Peak Slices",
            'coelu': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Peak Coelu Slices",
            'coelu csv': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Peak Coelu CSV",
            'coelu sliced': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Coelu Slices Sliced",
            'patch': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Peak Patches",
            'pixel': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Pixel CSVs",
            'counts': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Peak Counts",
            'composites': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Composites",
            'clustering': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Clustering",
            'noncoelu': cls.BASE_DIR / cls.OUTPUT_ROOT / cls.ANALYSIS_FOLDER / cls.CURRENT_GROUP / "Non Coelu Slices"
        }

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        return {k: str(v) for k, v in dirs.items()}