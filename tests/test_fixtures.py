import tempfile
import unittest
import zipfile
from pathlib import Path

from uiir.fixtures import FIXTURE_PRESETS, FixtureSource, download_fixture_set


class FixtureDownloadTests(unittest.TestCase):
    def test_downloads_direct_psd_and_zip_manifest_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_psd = root / "source.psd"
            source_psd.write_bytes(b"psd")
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("nested/source.psd", b"nested-psd")

            FIXTURE_PRESETS["unit-direct"] = [
                FixtureSource(
                    "direct",
                    url=source_psd.as_uri(),
                    file_name="direct.psd",
                    license="CC0",
                    source_url="https://example.test/direct",
                    attribution="Example",
                ),
                FixtureSource(
                    "zip",
                    url=archive.as_uri(),
                    file_name="archive.zip",
                    license="CC0",
                    source_url="https://example.test/zip",
                    attribution="Example Zip",
                ),
            ]
            try:
                manifest = download_fixture_set("unit-direct", root / "fixtures")
            finally:
                FIXTURE_PRESETS.pop("unit-direct", None)

            self.assertEqual(manifest["count"], 2)
            self.assertTrue((root / "fixtures" / "direct" / "direct.psd").exists())
            self.assertTrue((root / "fixtures" / "zip" / "extracted" / "nested" / "source.psd").exists())
            for item in manifest["files"]:
                self.assertEqual(item["license"], "CC0")
                self.assertEqual(item["expected"], "ok")
                self.assertTrue(item["sha256"])


if __name__ == "__main__":
    unittest.main()
