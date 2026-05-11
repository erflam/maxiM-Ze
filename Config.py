from pathlib import Path
import csv
import json

class Config:
    BASE_DIR = Path.home() / "Desktop/maxiMiZe Tests"
    INPUT_SUBDIR = Path("maxiMiZe Files")
    OUTPUT_ROOT = Path("")

    # ── Run name ──────────────────────────────────────────────────────
    # Set by the GUI via Config.ANALYSIS_FOLDER = <user input>.
    # Defaults to an empty string so the output root is BASE_DIR/OUTPUT_ROOT.
    ANALYSIS_FOLDER = "maxiMZe Run"

    USE_DYNAMIC_MASS_GROUPS = True         # MUST be True (fallback groups removed)
    MAX_GROUPS_TO_RUN = 30                # None = all, N = first N groups
    REBUILD_MASS_GROUPS = True            # True = ignore cache and recompute
    GROUPING_VERBOSE = False               # True = print per-file processing during grouping

    GROUP_NOISE_LEVEL = 5000.0
    GROUP_MZ_TOLERANCE = 0.0005
    GROUP_MIN_CONSEC_SCANS = 7
    GROUP_MIN_SAMPLE_PRESENCE = 1
    GROUP_MIN_GROUP_SIZE = 2
    GROUP_MAX_GROUP_SIZE = 5

    MASS_GROUPS_CACHE_NAME = "MassGroups_Cache.json"
    MASS_GROUPS_EXPORT_NAME = "MassGroups_Formatted.csv"
    MASS_GROUPS: dict[str, list[float]] = {}   # populated by initialize_mass_groups() or cache
    MASS_GROUPS_INTENSITIES: dict[str, list[float]] = {}  # parallel intensities for CSV export
    MASS_GROUPS_RTS: dict[str, list[float | None]] = {}  # parallel RT apex values for CSV export
    MASS_GROUPS_RT_STARTS: dict[str, list[float | None]] = {}  # parallel RT start values
    MASS_GROUPS_RT_ENDS: dict[str, list[float | None]] = {}  # parallel RT end values

    # Easy grouping diagnostics/tuning
    GROUP_RT_APEX_SEPARATION = 1.0       # candidates closer than this RT distance are not grouped
    GROUP_MIN_INTENSITY_RATIO = 0.55     # MUST stay 0.55 per your requirement
    GROUP_REQUIRE_NO_RT_OVERLAP = True   # True = RT windows cannot overlap
    MASS_LIST: list[float] = []               # active group's masses
    CURRENT_GROUP: str | None = None
    MASS_TOLERANCE = 0.0005
    MAX_PEAK_DURATION = 1.5

    # Run specific masses only
    RUN_ONLY_MASSES = None  # [247.1540] or set to None to run normal dynamic groups

    # Target files (pooled controls) — set by GUI when use_target is enabled.
    # When non-empty, MassGrouping.select_files() guarantees one is picked.
    TARGET_FILES: list[str] = []

    # Library Matching
    LIB_FILE = r"/Users/elizabethflammer/Desktop/Research/MZMine/POS OE Library Metformin Baseline.csv"
    LIBRARY_MATCH_MZ_TOL = 0.0005
    LIBRARY_MATCH_RT_TOL = 0.125

    @classmethod
    def _analysis_output_root(cls) -> Path:
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

    # ── Environment variable used to pass cache path to spawned workers ──
    _CACHE_PATH_ENV = "MAXIMIZE_CACHE_PATH"

    @classmethod
    def save_mass_groups_cache(cls) -> None:
        """
        Write mass groups to disk and record the path in an env var so spawned
        worker processes can locate the same cache.

        This cache supports both:
          - old format: {"Group1": [mz1, mz2]}
          - new rich format with masses/intensities/RTs
        """
        import os
        p = cls._mass_groups_cache_path()

        cache_data = {}
        for name, masses in cls.MASS_GROUPS.items():
            cache_data[name] = {
                "masses": [float(x) for x in masses],
                "intensities": [float(x) for x in cls.MASS_GROUPS_INTENSITIES.get(name, [])],
                "rts": [
                    None if x is None else float(x)
                    for x in cls.MASS_GROUPS_RTS.get(name, [])
                ],
                "rt_starts": [
                    None if x is None else float(x)
                    for x in cls.MASS_GROUPS_RT_STARTS.get(name, [])
                ],
                "rt_ends": [
                    None if x is None else float(x)
                    for x in cls.MASS_GROUPS_RT_ENDS.get(name, [])
                ],
            }

        p.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        os.environ[cls._CACHE_PATH_ENV] = str(p)
        print(f"[Config] Mass groups cache saved → {p}")

    @classmethod
    def _resolve_cache_path(cls) -> Path:
        """
        Return the cache path to use. Workers get it from the env var written
        by the main process; the main process derives it from class variables.
        """
        import os
        env_path = os.environ.get(cls._CACHE_PATH_ENV)
        if env_path:
            return Path(env_path)
        return cls._mass_groups_cache_path()

    @classmethod
    def load_mass_groups_cache(cls) -> bool:
        p = cls._resolve_cache_path()
        if not p.exists():
            return False

        data = json.loads(p.read_text(encoding="utf-8"))

        cls.MASS_GROUPS = {}
        cls.MASS_GROUPS_INTENSITIES = {}
        cls.MASS_GROUPS_RTS = {}
        cls.MASS_GROUPS_RT_STARTS = {}
        cls.MASS_GROUPS_RT_ENDS = {}

        for name, value in data.items():
            # Backward compatible old format: "Group1": [166.0862, 116.0706]
            if isinstance(value, list):
                cls.MASS_GROUPS[name] = [float(x) for x in value]
                continue

            # New rich format
            cls.MASS_GROUPS[name] = [float(x) for x in value.get("masses", [])]
            cls.MASS_GROUPS_INTENSITIES[name] = [float(x) for x in value.get("intensities", [])]
            cls.MASS_GROUPS_RTS[name] = [
                None if x is None else float(x)
                for x in value.get("rts", [])
            ]
            cls.MASS_GROUPS_RT_STARTS[name] = [
                None if x is None else float(x)
                for x in value.get("rt_starts", [])
            ]
            cls.MASS_GROUPS_RT_ENDS[name] = [
                None if x is None else float(x)
                for x in value.get("rt_ends", [])
            ]

        return True

    @classmethod
    def load_mass_groups_from_json(cls, json_path: str | Path) -> bool:
        """
        Load MASS_GROUPS from an external JSON file supplied by the user.
        Supports both old list-only cache and new rich cache with RTs.
        """
        p = Path(json_path)
        if not p.exists():
            return False

        # Temporarily point loader at the imported JSON
        import os
        old_env = os.environ.get(cls._CACHE_PATH_ENV)
        os.environ[cls._CACHE_PATH_ENV] = str(p)
        ok = cls.load_mass_groups_cache()
        if old_env is None:
            os.environ.pop(cls._CACHE_PATH_ENV, None)
        else:
            os.environ[cls._CACHE_PATH_ENV] = old_env

        if ok:
            cls.save_mass_groups_cache()
        return ok

    @classmethod
    def ensure_mass_groups_loaded(cls) -> None:
        """
        Used by multiprocessing workers.
        Spawned workers re-import Config from scratch, so class variables set
        at runtime in the parent process are lost.  We locate the cache via
        the MAXIMIZE_CACHE_PATH env var, which is inherited by child processes.
        """
        if cls.MASS_GROUPS:
            return
        if not cls.load_mass_groups_cache():
            cache_path = cls._resolve_cache_path()
            raise RuntimeError(
                "MASS_GROUPS not initialized in this process and cache not found.\n"
                f"Expected cache at: {cache_path}\n"
                "Fix: make sure the main process calls Config.initialize_mass_groups(...) "
                "before starting multiprocessing."
            )

    @classmethod
    def initialize_mass_groups(
        cls,
        all_input_files: list[str] | None = None,
        *,
        import_json_path: str | Path | None = None,
    ) -> None:
        """
        Builds or loads MASS_GROUPS, then writes the cache so workers can find it.

        Parameters
        ----------
        import_json_path
            If provided (and the file exists), load groups from this JSON and skip
            detection entirely.  This is the "import previous JSON" option from the GUI.
        """

        # ── Option 0: caller supplied a previous JSON cache ───────────
        if import_json_path is not None:
            p = Path(import_json_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Imported JSON cache not found: {p}\n"
                    "Make sure you selected the correct MassGroups_Cache.json file."
                )
            print(f"[Config] Loading mass groups from imported JSON: {p}")
            if not cls.load_mass_groups_from_json(p):
                raise ValueError(f"Failed to parse mass groups from: {p}")
            first_group = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)[0]
            cls.set_mass_group(first_group)
            cls.save_mass_groups_formatted_csv()
            return

        # ── Option 1: run-specific masses override everything ─────────
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

        # ── Option 2: already in memory and not forcing rebuild ────────
        if cls.MASS_GROUPS and not cls.REBUILD_MASS_GROUPS:
            cls.save_mass_groups_cache()          # ensure workers can find it
            cls.save_mass_groups_formatted_csv()
            return

        # ── Option 3: load from this run's own cache ───────────────────
        if not cls.REBUILD_MASS_GROUPS and cls.load_mass_groups_cache():
            first_group = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)[0]
            cls.set_mass_group(first_group)
            cls.save_mass_groups_formatted_csv()
            return

        # ── Option 4: build fresh (slow path) ─────────────────────────
        from MassGrouping import select_files, build_mass_groups_from_files

        print("[Config] Detecting mass groups fresh (this may take a few minutes)...")
        grouping_files = select_files(
            all_input_files=all_input_files,
            target_files=cls.TARGET_FILES or [],
        )

        grouping_result = build_mass_groups_from_files(
            grouping_files,
            noise_level=cls.GROUP_NOISE_LEVEL,
            mz_tolerance=cls.GROUP_MZ_TOLERANCE,
            min_consec_scans=cls.GROUP_MIN_CONSEC_SCANS,
            min_sample_presence=cls.GROUP_MIN_SAMPLE_PRESENCE,
            min_group_size=cls.GROUP_MIN_GROUP_SIZE,
            max_group_size=cls.GROUP_MAX_GROUP_SIZE,
            verbose=getattr(cls, "GROUPING_VERBOSE", False),
        )

        # Supports either:
        #   old MassGrouping return: (groups, intensities)
        #   new MassGrouping return: (groups, intensities, rts, rt_starts, rt_ends)
        if len(grouping_result) == 2:
            cls.MASS_GROUPS, cls.MASS_GROUPS_INTENSITIES = grouping_result
            cls.MASS_GROUPS_RTS = {}
            cls.MASS_GROUPS_RT_STARTS = {}
            cls.MASS_GROUPS_RT_ENDS = {}
            print("[Config] WARNING: MassGrouping returned only masses/intensities, so RT columns will be blank.")
            print("[Config] Update MassGrouping.py to return RT dictionaries; see patch below.")
        else:
            (
                cls.MASS_GROUPS,
                cls.MASS_GROUPS_INTENSITIES,
                cls.MASS_GROUPS_RTS,
                cls.MASS_GROUPS_RT_STARTS,
                cls.MASS_GROUPS_RT_ENDS,
            ) = grouping_result

        if not cls.MASS_GROUPS:
            raise ValueError("Dynamic mass grouping produced 0 groups.")

        first_group = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)[0]
        cls.set_mass_group(first_group)

        # CRITICAL: write cache BEFORE any multiprocessing workers are spawned
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
        Saves ALL groups to MassGroups_Formatted.csv.

        Output shape:
          Group, Size, Masses, Intensities,
          Mass1, RT1, Intensity1, Mass2, RT2, Intensity2, ...

        RT columns will populate only when MassGrouping.py returns RT dictionaries.
        """
        cls.ensure_mass_groups_loaded()

        out_dir = cls._analysis_output_root()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cls.MASS_GROUPS_EXPORT_NAME

        group_names = sorted(cls.MASS_GROUPS.keys(), key=cls._group_sort_key)
        max_size = max((len(cls.MASS_GROUPS.get(name, [])) for name in group_names), default=0)

        header = ["Group", "Size", "Masses", "Intensities"]
        for i in range(1, max_size + 1):
            header.extend([f"Mass{i}", f"RT{i}", f"Intensity{i}"])

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for name in group_names:
                masses = cls.MASS_GROUPS.get(name, [])
                intensities = cls.MASS_GROUPS_INTENSITIES.get(name, [])
                rts = cls.MASS_GROUPS_RTS.get(name, [])

                masses_str = "[" + ", ".join(f"{float(m):.4f}" for m in masses) + "]"
                intensities_str = (
                    "[" + ", ".join(f"{float(x):.0f}" for x in intensities) + "]"
                    if intensities else "N/A"
                )

                row = [cls._display_group_label(name), len(masses), masses_str, intensities_str]

                for i in range(max_size):
                    mass = masses[i] if i < len(masses) else ""
                    intensity = intensities[i] if i < len(intensities) else ""
                    rt = rts[i] if i < len(rts) else ""

                    row.extend([
                        "" if mass == "" else f"{float(mass):.4f}",
                        "" if rt in ("", None) else f"{float(rt):.4f}",
                        "" if intensity == "" else f"{float(intensity):.0f}",
                    ])

                writer.writerow(row)

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
