from Config import Config
import random

class FileUtils:
    @staticmethod
    def get_file_paths():
        """Get list of input file paths."""
        file_names = [
            "OE_EF_IsmailBaseline_POS_C072_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C072_0004.mzXML",
            "OE_EF_IsmailBaseline_POS_C072_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C068_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C065_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C065_0004.mzXML",
            "OE_EF_IsmailBaseline_POS_C065_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C063_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C062_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C062_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C057_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C048_0008.mzXML",
            "OE_EF_IsmailBaseline_POS_C048_0004.mzXML",
            "OE_EF_IsmailBaseline_POS_C047_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C046_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C039_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C039_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C039_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C037_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C034_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C034_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C034_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C032_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C032_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C031_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C031_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C031_0003.mzXML",
            "OE_EF_IsmailBaseline_POS_C020_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C020_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C020_0004.mzXML",
            "OE_EF_IsmailBaseline_POS_C020_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C018_0012.mzXML",
            "OE_EF_IsmailBaseline_POS_C018_0010.mzXML",
            "OE_EF_IsmailBaseline_POS_C016_0007.mzXML",
            "OE_EF_IsmailBaseline_POS_C016_0006.mzXML",
            "OE_EF_IsmailBaseline_POS_C016_0004.mzXML",
            "OE_EF_IsmailBaseline_POS_C016_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C014_0007.mzXML",
            "OE_EF_IsmailBaseline_POS_C014_0005.mzXML",
            "OE_EF_IsmailBaseline_POS_C014_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C012_0001.mzXML",
            "OE_EF_IsmailBaseline_POS_C009_0002.mzXML",
            "OE_EF_IsmailBaseline_POS_C007_0002.mzXML"
        ]
        return [str(Config.BASE_DIR / Config.INPUT_SUBDIR / fn) for fn in file_names]

    @staticmethod
    def random_dark_hex_color():
        """Generate random dark color for plotting."""
        return "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100)
        )
