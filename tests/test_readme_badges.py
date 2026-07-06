import unittest
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


class ReadmeBadgeTests(unittest.TestCase):
    def test_huggingface_downloads_badge_is_dynamic(self):
        readme = README.read_text(encoding="utf-8")

        self.assertNotRegex(readme, r"downloads%2Fmonth-\d+")
        self.assertIn("img.shields.io/badge/dynamic/json", readme)
        self.assertIn("huggingface.co%2Fapi%2Fdatasets%2FsharryXR%2FGUIDE-dataset%3Fexpand%3Ddownloads", readme)
        self.assertIn("query=%24.downloads", readme)
        self.assertIn("label=downloads%2Fmonth", readme)


if __name__ == "__main__":
    unittest.main()
