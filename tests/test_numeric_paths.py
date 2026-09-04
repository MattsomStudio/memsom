"""Numeric fast paths and the thread rules around heavy imports (2026-09-04).

Why these exist — MEASURED on the PC, py-spy on the live MCP server and a
self-spawned repro:

* While any thread sits in a blocking stdin pipe read (the MCP server's main
  loop, always), a FIRST C-extension import on another thread blocks until that
  read returns. The warm-endpoint request thread hung for minutes importing
  numpy (colbert_maxsim) and then sklearn/scipy (bge_available -> FlagEmbedding),
  so every prompt-hook query timed out, backed off, and read as
  "warm endpoint down" with a healthy encoder.
* Even warm, the pure-Python dense cosine (0.29 s / 757 rows) and the
  struct.unpack -> list -> asarray ColBERT decode (1.6 s / 100 candidates) blew
  the hook's warm budget on their own.

Contracts held here:
  1. numpy_for_scoring never imports off the main thread; on the main thread it
     does (when installed).
  2. bge_available is a find_spec probe: it never pulls torch/FlagEmbedding in.
  3. _get_model refuses a FIRST FlagEmbedding import off the main thread.
  4. The numpy dense / ColBERT paths give the same scores as the pure ones.
  5. mcp.preload_numeric loads numpy on the main thread and nothing heavier.
"""
import json
import struct
import subprocess
import sys
import threading
import unittest
from unittest import mock

from memsom.retrieval import embed as memsom_embed
from memsom.retrieval import retrieve as memsom_retrieve

HAVE_NUMPY = memsom_embed.numpy_for_scoring() is not None


def _in_thread(fn):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # test plumbing: surface the exception to the caller
            box["exc"] = exc

    th = threading.Thread(target=run)
    th.start()
    th.join(30)
    if "exc" in box:
        raise box["exc"]
    return box["value"]


class TestNumpyThreadRule(unittest.TestCase):
    def test_off_main_thread_with_numpy_absent_returns_none_and_does_not_import(self):
        # sys.modules["numpy"] = None makes a real `import numpy` raise; the
        # rule must return None BEFORE ever getting there.
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with mock.patch("importlib.import_module") as imp:
                self.assertIsNone(_in_thread(memsom_embed.numpy_for_scoring))
                imp.assert_not_called()
            self.assertFalse(_in_thread(memsom_embed.numpy_scoring_available))

    def test_import_blocked_only_when_installed_unloaded_and_off_main_thread(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            # installed (find_spec finds it) + not loaded + off main thread -> blocked
            with mock.patch("importlib.util.find_spec", return_value=object()):
                self.assertTrue(_in_thread(memsom_embed.numpy_import_blocked))
                self.assertFalse(memsom_embed.numpy_import_blocked())  # main thread
            # NOT installed (the CI box) -> never blocked: the pure path runs
            with mock.patch("importlib.util.find_spec", return_value=None):
                self.assertFalse(_in_thread(memsom_embed.numpy_import_blocked))
        with mock.patch.dict(sys.modules, {"numpy": object()}):
            self.assertFalse(_in_thread(memsom_embed.numpy_import_blocked))

    def test_already_loaded_numpy_is_used_from_any_thread(self):
        sentinel = object()
        with mock.patch.dict(sys.modules, {"numpy": sentinel}):
            self.assertIs(_in_thread(memsom_embed.numpy_for_scoring), sentinel)
            self.assertIs(memsom_embed.numpy_for_scoring(), sentinel)

    def test_main_thread_imports_when_absent(self):
        # A fresh process: numpy is absent until the main thread asks for it.
        # (numpy's C extension cannot be re-imported in-process, so this is
        # the only honest way to test the first-import branch.)
        code = (
            "import sys, json\n"
            "from memsom.retrieval import embed as e\n"
            "before = 'numpy' in sys.modules\n"
            "mod = e.numpy_for_scoring()\n"
            "print(json.dumps({'before': before, 'loaded': mod is not None,"
            " 'name': getattr(mod, '__name__', None), 'after': 'numpy' in sys.modules}))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertFalse(data["before"], "importing memsom.retrieval.embed must not load numpy")
        if HAVE_NUMPY:
            self.assertTrue(data["loaded"])
            self.assertEqual(data["name"], "numpy")
            self.assertTrue(data["after"])
        else:
            self.assertFalse(data["loaded"])


class TestBgeAvailableIsAProbe(unittest.TestCase):
    def test_bge_available_never_imports_torch(self):
        code = (
            "import sys, json\n"
            "from memsom.retrieval import embed as e\n"
            "e._BGE_AVAILABLE = None\n"
            "ok = e.bge_available()\n"
            "print(json.dumps({'ok': ok, 'torch': 'torch' in sys.modules,"
            " 'flag': 'FlagEmbedding' in sys.modules, 'numpy': 'numpy' in sys.modules}))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertIsInstance(data["ok"], bool)
        self.assertFalse(data["torch"], "bge_available imported torch")
        self.assertFalse(data["flag"], "bge_available imported FlagEmbedding")

    def test_get_model_refuses_first_import_off_main_thread(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("FlagEmbedding", None)
            memsom_embed._MODEL = None
            with self.assertRaises(RuntimeError):
                _in_thread(memsom_embed._get_model)
            self.assertNotIn("FlagEmbedding", sys.modules)


def _colbert_blob(matrix):
    flat = [x for row in matrix for x in row]
    return struct.pack(f"<{len(flat)}e", *flat)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed: pure paths only")
class TestNumpyPathsMatchPure(unittest.TestCase):
    def test_dense_scores_match_cosine_loop(self):
        import random
        rnd = random.Random(7)
        dim = 16
        q = [rnd.uniform(-1, 1) for _ in range(dim)]
        rows = []
        for nid in range(1, 12):
            vec = [rnd.uniform(-1, 1) for _ in range(dim)]
            rows.append((nid, memsom_retrieve._vec_to_blob(vec)))
        rows.append((99, memsom_retrieve._vec_to_blob([0.0] * dim)))     # zero vector
        rows.append((100, memsom_retrieve._vec_to_blob([1.0] * 4)))      # wrong dim
        fast = dict(memsom_retrieve._dense_scores(q, rows))
        pure = {nid: memsom_retrieve._cosine(q, memsom_retrieve._blob_to_vec(blob))
                for nid, blob in rows}
        self.assertEqual(set(fast), set(pure))
        for nid in pure:
            self.assertAlmostEqual(fast[nid], pure[nid], places=5, msg=nid)
        self.assertEqual(fast[99], 0.0)
        self.assertEqual(fast[100], 0.0)

    def test_colbert_blob_path_matches_list_path(self):
        import random
        rnd = random.Random(3)
        dim = 8
        q = [[rnd.uniform(-1, 1) for _ in range(dim)] for _ in range(5)]
        d = [[rnd.uniform(-1, 1) for _ in range(dim)] for _ in range(7)]
        blob = _colbert_blob(d)
        np = memsom_embed.numpy_for_scoring()
        q_np = np.asarray(q, dtype=np.float32)
        fast = memsom_embed.colbert_maxsim_blob(q_np, blob, 7, dim)
        pure = memsom_embed.colbert_maxsim(q, memsom_embed.blob_to_colbert(blob, 7, dim))
        self.assertIsNotNone(fast)
        self.assertAlmostEqual(fast, pure, places=3)
        # shape mismatch -> None (caller falls back), never a wrong score
        self.assertIsNone(memsom_embed.colbert_maxsim_blob(q_np, blob, 6, dim))
        self.assertIsNone(memsom_embed.colbert_maxsim_blob(q_np, blob, 7, 0))

    def test_colbert_maxsim_off_main_thread_without_numpy_uses_pure_path(self):
        q = [[1.0, 0.0], [0.0, 1.0]]
        d = [[0.5, 0.5], [1.0, 0.0]]
        with mock.patch.dict(sys.modules, {"numpy": None}):
            score = _in_thread(lambda: memsom_embed.colbert_maxsim(q, d))
        self.assertAlmostEqual(score, 1.0 + 0.5, places=6)


class TestMcpPreload(unittest.TestCase):
    def test_preload_loads_numpy_only(self):
        code = (
            "import sys, json\n"
            "from memsom.interface import mcp\n"
            "ok = mcp.preload_numeric()\n"
            "print(json.dumps({'ok': ok, 'numpy': 'numpy' in sys.modules,"
            " 'torch': 'torch' in sys.modules, 'flag': 'FlagEmbedding' in sys.modules,"
            " 'sklearn': 'sklearn' in sys.modules}))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(data["ok"], data["numpy"])
        self.assertFalse(data["torch"])
        self.assertFalse(data["flag"])
        self.assertFalse(data["sklearn"])


if __name__ == "__main__":
    unittest.main()
