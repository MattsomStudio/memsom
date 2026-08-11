#!/usr/bin/env python3
"""Tests for memsom.kernel.syncguard -- file-sync tree detection (PLAN.md Sec3.4)."""

import tempfile
import unittest
from pathlib import Path

from memsom.kernel import syncguard as S


class TestClean(unittest.TestCase):
    def test_plain_tmpdir_is_clean(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(S.sync_markers(Path(td) / "store"), [])


class TestSyncthing(unittest.TestCase):
    def test_stfolder_marker_detected_on_ancestor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".stfolder").touch()
            nested = root / "sub" / "store"
            found = S.sync_markers(nested)
            self.assertTrue(any("Syncthing" in f for f in found))

    def test_stignore_and_stversions_also_detected(self):
        for marker in (".stignore", ".stversions"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / marker).touch()
                found = S.sync_markers(root / "store")
                self.assertTrue(any(marker in f for f in found), marker)


class TestDropbox(unittest.TestCase):
    def test_dropbox_marker_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".dropbox").touch()
            found = S.sync_markers(root / "a" / "b" / "store")
            self.assertTrue(any("Dropbox" in f for f in found))


class TestGoogleDrive(unittest.TestCase):
    def test_tmp_driveupload_marker_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".tmp.driveupload").touch()
            found = S.sync_markers(root / "store")
            self.assertTrue(any("Google Drive" in f for f in found))

    def test_drivefs_dir_name_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "DriveFS"
            root.mkdir()
            found = S.sync_markers(root / "store")
            self.assertTrue(any("Google Drive" in f for f in found))


class TestICloud(unittest.TestCase):
    def test_mobile_documents_dir_name_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Mobile Documents"
            root.mkdir()
            found = S.sync_markers(root / "store")
            self.assertTrue(any("iCloud" in f for f in found))


class TestOneDrive(unittest.TestCase):
    def test_onedrive_prefix_match_detected(self):
        with tempfile.TemporaryDirectory() as td:
            onedrive_root = Path(td) / "OneDrive"
            store = onedrive_root / "sub" / "store"
            found = S.sync_markers(store, environ={"OneDrive": str(onedrive_root)})
            self.assertTrue(any("OneDrive" in f for f in found))

    def test_path_outside_onedrive_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            onedrive_root = Path(td) / "OneDrive"
            other = Path(td) / "elsewhere" / "store"
            found = S.sync_markers(other, environ={"OneDrive": str(onedrive_root)})
            self.assertEqual(found, [])

    def test_onedrive_commercial_variant(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "OneDrive-Corp"
            store = root / "store"
            found = S.sync_markers(store, environ={"OneDriveCommercial": str(root)})
            self.assertTrue(any("OneDrive" in f for f in found))


class TestExtraMarkers(unittest.TestCase):
    def test_custom_marker_env_knob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".customsync").touch()
            found = S.sync_markers(root / "store",
                                   environ={"MEMSOM_EXTRA_SYNC_MARKERS": ".customsync,.other"})
            self.assertTrue(any("custom marker" in f for f in found))

    def test_extra_markers_knob_empty_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            found = S.sync_markers(Path(td) / "store", environ={})
            self.assertEqual(found, [])


class TestDbIntegration(unittest.TestCase):
    """storage.db.get_connection() refuses a synced store at run time --
    the twin of the setup-time check (Sec3.4: 'enforced, in both places')."""

    def test_get_connection_refuses_synced_store(self):
        import os
        import memsom
        from memsom.kernel.syncguard import SyncGuardError

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".stfolder").touch()
            db = root / "memdag.db"
            old_home, old_db = os.environ.get("MEMDAG_HOME"), os.environ.get("MEMDAG_DB")
            os.environ["MEMDAG_HOME"] = str(root)
            os.environ["MEMDAG_DB"] = str(db)
            try:
                with self.assertRaises(SyncGuardError):
                    memsom.get_connection()
            finally:
                if old_home is None:
                    os.environ.pop("MEMDAG_HOME", None)
                else:
                    os.environ["MEMDAG_HOME"] = old_home
                if old_db is None:
                    os.environ.pop("MEMDAG_DB", None)
                else:
                    os.environ["MEMDAG_DB"] = old_db

    def test_get_connection_allows_synced_store_when_acknowledged(self):
        import json
        import os
        import memsom

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".stfolder").touch()
            (root / "memsom.json").write_text(
                json.dumps({"sync_check": "acknowledged-unsafe"}), encoding="utf-8")
            db = root / "memdag.db"
            old_home, old_db = os.environ.get("MEMDAG_HOME"), os.environ.get("MEMDAG_DB")
            os.environ["MEMDAG_HOME"] = str(root)
            os.environ["MEMDAG_DB"] = str(db)
            try:
                conn = memsom.get_connection()
                conn.close()
            finally:
                if old_home is None:
                    os.environ.pop("MEMDAG_HOME", None)
                else:
                    os.environ["MEMDAG_HOME"] = old_home
                if old_db is None:
                    os.environ.pop("MEMDAG_DB", None)
                else:
                    os.environ["MEMDAG_DB"] = old_db

    def test_read_only_connection_skips_sync_check(self):
        import os
        import memsom

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".stfolder").touch()
            db = root / "memdag.db"
            old_home, old_db = os.environ.get("MEMDAG_HOME"), os.environ.get("MEMDAG_DB")
            os.environ["MEMDAG_HOME"] = str(root)
            os.environ["MEMDAG_DB"] = str(db)
            try:
                with self.assertRaises(FileNotFoundError):
                    memsom.get_connection(read_only=True)
            finally:
                if old_home is None:
                    os.environ.pop("MEMDAG_HOME", None)
                else:
                    os.environ["MEMDAG_HOME"] = old_home
                if old_db is None:
                    os.environ.pop("MEMDAG_DB", None)
                else:
                    os.environ["MEMDAG_DB"] = old_db


if __name__ == "__main__":
    unittest.main()
