from pathlib import Path

class Config:
    # Base directories
    BASE_DIR = Path.home() / "Desktop/maxiMiZe Tests"
    INPUT_SUBDIR = Path("maxiMiZe Files")
    OUTPUT_ROOT = Path("maxiMiZe Checkpoints")
    ANALYSIS_FOLDER = ("maxiMZe Group 1-10 0223 Test 1")

    # Analysis parameters
    MASS_GROUPS = {
        'Group1': [169.0356, 182.0811, 297.1672, 132.1019],
        'Group2': [247.1540, 104.1069, 167.0896],
        'Group3': [104.0706, 86.0964, 269.1358],
        'Group4': [206.1005, 393.2859, 233.1383],
        'Group5': [235.1652, 187.0964, 261.1697],
        'Group6': [169.0583, 232.1544, 274.2741],
        'Group7': [119.0896, 337.0641, 247.1441],
        'Group8': [70.0651, 283.1515, 175.1077],
        'Group9': [179.0484, 409.1871, 292.2119],
        'Group10': [158.9640, 314.2327, 280.1391],
    }
    MASS_LIST = MASS_GROUPS["Group1"]  # Default to first group for backwards compatibility
    MASS_TOLERANCE = 0.0005
    MAX_PEAK_DURATION = 1.5
    CURRENT_GROUP = "Group1"  # Track current group being processed

    @classmethod
    def set_mass_group(cls, group_name):
        """Set the current mass group to process."""
        if group_name not in cls.MASS_GROUPS:
            raise ValueError(f"Invalid group name: {group_name}")
        cls.CURRENT_GROUP = group_name
        cls.MASS_LIST = cls.MASS_GROUPS[group_name]

    @classmethod
    def setup_directories(cls):
        """Setup and return all required output directories."""
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