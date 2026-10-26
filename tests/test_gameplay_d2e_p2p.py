"""D2E (OWAMcap) and Open-P2P (protobuf) adapters.

Fixtures are synthetic but use the field names and encodings read from the
public samples (open-world-agents/D2E-480p, elefantai/p2p-toy-examples); no
dataset files are shipped. Set EMBODIGEN_D2E_SAMPLES / EMBODIGEN_P2P_SAMPLES
to a downloaded folder to also run the adapters on real files.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.pipeline import curate, run_ingest
from embodigen.gameplay.recording import read_recording
from embodigen.gameplay.registry import WorldRegistry
from embodigen.gameplay.sources import protowire as pw
from embodigen.gameplay.sources.d2e import short_stem, vk_name
from embodigen.gameplay.sources.mcap import McapReader, write_mcap
from embodigen.gameplay.sources.p2p import human_runs
from embodigen.gameplay.splits import SplitConfig, split
from embodigen.gameplay.types import BUTTON, FOCUS, KEY, MOVE, WHEEL
from embodigen.gameplay.windows import WindowConfig, make_windows

try:
    from compression import zstd  # noqa: F401

    HAS_ZSTD = True
except ImportError:
    try:
        import zstandard  # noqa: F401

        HAS_ZSTD = True
    except ImportError:
        HAS_ZSTD = False

NS = 1_000_000_000
REGISTRY = WorldRegistry.load()


def d2e_messages(stem: str, seconds: float = 30.0, alt_tab: tuple[float, float] | None = None) -> list:
    """A D2E-shaped session: 60 fps screen, 1 kHz raw mouse, W held with
    Windows auto-repeat, clicks, a wheel notch and a 1 Hz window topic."""
    msgs = []

    def add(topic, schema, t, payload):
        msgs.append((topic, schema, int(round(t * NS)), json.dumps(payload).encode()))

    for k in range(int(seconds * 60)):
        t = 0.02 + k / 60
        add("screen", "desktop/ScreenCaptured", t, {
            "utc_ns": 1756239829270489800 + int(t * NS), "source_shape": [1920, 1080], "shape": [1920, 1080],
            "media_ref": {"uri": f"{stem}.mkv", "pts_ns": int(t * NS)}})
    for k in range(int(seconds * 1000)):
        t = k / 1000
        if int(t) % 3 == 0:  # turn for one second out of three
            add("mouse/raw", "desktop/RawMouseEvent", t, {
                "last_x": 2, "last_y": -1, "us_flags": 0, "button_flags": 0, "button_data": 0, "device_handle": 65593})
    for s in range(int(seconds)):
        add("keyboard", "desktop/KeyboardEvent", s + 0.1, {"event_type": "press", "vk": 87})
        for r in range(1, 10):  # auto-repeat while held
            add("keyboard", "desktop/KeyboardEvent", s + 0.1 + 0.05 * r, {"event_type": "press", "vk": 87})
        add("keyboard", "desktop/KeyboardEvent", s + 0.8, {"event_type": "release", "vk": 87})
        add("keyboard", "desktop/KeyboardEvent", s + 0.3, {"event_type": "press", "vk": 0xA0})
        add("keyboard", "desktop/KeyboardEvent", s + 0.4, {"event_type": "release", "vk": 0xA0})
        add("mouse/raw", "desktop/RawMouseEvent", s + 0.5, {"last_x": 0, "last_y": 0, "us_flags": 0, "button_flags": 1, "button_data": 0})
        add("mouse/raw", "desktop/RawMouseEvent", s + 0.6, {"last_x": 0, "last_y": 0, "us_flags": 0, "button_flags": 2, "button_data": 0})
        title = "Minecraft 1.21.8 - Singleplayer"
        if alt_tab and alt_tab[0] <= s < alt_tab[1]:
            title = "Discord"
        add("window", "desktop/WindowInfo", s + 0.9, {"title": title, "rect": [0, 0, 1920, 1080], "hWnd": 1})
    add("mouse/raw", "desktop/RawMouseEvent", 2.25, {"last_x": 0, "last_y": 0, "us_flags": 0,
                                                    "button_flags": 0x400, "button_data": 0xFF88})  # -120: one notch down
    msgs.sort(key=lambda m: m[2])
    return msgs


def p2p_annotation(frames: list[str], env: str = "roblox", subtype: str = "blade-ball") -> bytes:
    """A serialized VideoAnnotation: ``frames`` is 'user' / 'system' / 'unknown' per frame."""
    window = pw.varint_field(1, 1920) + pw.varint_field(2, 1080) + pw.double_field(3, 1.0)
    specs = pw.bytes_field(1, "windows") + pw.bytes_field(3, window)
    meta = (pw.bytes_field(1, "0199-test") + pw.bytes_field(3, "player7") + pw.float_field(5, 20.0)
            + pw.bytes_field(6, pw.bytes_field(1, env) + pw.bytes_field(2, subtype)) + pw.bytes_field(10, specs))
    out = pw.bytes_field(2, meta)
    for i, who in enumerate(frames):
        keys = pw.bytes_field(1, "w") + (pw.bytes_field(1, "LeftShift") if i % 20 < 5 else b"")
        delta = pw.varint_field(1, -7) + pw.varint_field(2, 3)
        mouse = pw.bytes_field(1, pw.varint_field(1, 900) + pw.varint_field(2, 500)) + pw.bytes_field(3, delta)
        if i % 40 < 4:
            mouse += pw.bytes_field(9, "0")
        action = pw.bytes_field(2, keys) + pw.varint_field(3, 1) + pw.bytes_field(4, mouse)
        frame = b""
        if who == "user":
            frame += pw.bytes_field(1, action)
        elif who == "system":
            frame += pw.bytes_field(1, pw.varint_field(3, 0)) + pw.bytes_field(6, action)
        if i == 3:
            frame += pw.bytes_field(7, pw.bytes_field(1, "cross the bridge") + pw.float_field(3, 2.0))
        out += pw.bytes_field(3, frame)
    return out


class McapReaderTest(unittest.TestCase):
    def _roundtrip(self, chunk):
        msgs = [("keyboard", "desktop/KeyboardEvent", 5, b'{"event_type":"press","vk":87}'),
                ("screen", "desktop/ScreenCaptured", 9, b'{"utc_ns":1}')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.mcap"
            write_mcap(path, msgs, chunk=chunk)
            reader = McapReader(path)
            got = [(m.topic, m.schema, m.log_time, m.data) for m in reader.messages()]
            only = [m.topic for m in McapReader(path).messages({"screen"})]
        self.assertEqual(got, msgs)
        self.assertEqual(only, ["screen"])
        self.assertEqual(reader.profile, "owa")

    def test_top_level_and_chunked_messages(self):
        self._roundtrip(None)
        self._roundtrip("")

    @unittest.skipUnless(HAS_ZSTD, "zstd chunks need Python 3.14+ (or the [d2e] extra)")
    def test_zstd_chunks(self):
        self._roundtrip("zstd")


class D2ESourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        game = root / "d2e" / "Minecraft_1.21.8"
        game.mkdir(parents=True)
        for k, alt in ((0, (10, 14)), (1, None)):
            stem = f"recording_20250827_052328__366c9184_split_0{k}"
            write_mcap(game / f"{stem}.mcap", d2e_messages(stem, alt_tab=alt), chunk="")
            (game / f"{stem}.mkv").write_bytes(b"not really a video")
        cls.rows = run_ingest("d2e", root / "d2e", root / "raw", REGISTRY)
        cls.raw = root / "raw"
        cls.ledger = curate(cls.raw, root / "curated", workers=1)
        cls.curated = root / "curated"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rec(self, k):
        return read_recording(self.raw / f"d2e-minecraft-20250827_052328_366c9184_split_0{k}")

    def test_world_session_and_ids(self):
        self.assertEqual({r["world"] for r in self.rows}, {"minecraft"})
        self.assertEqual(len({r["session"] for r in self.rows}), 1)
        self.assertEqual(short_stem("recording_20250820_212408_Counter-Strike_2_ea30a554", "Counter-Strike_2"),
                         "20250820_212408_ea30a554")

    def test_keys_ignore_auto_repeat_and_use_canonical_names(self):
        rec = self.rec(1)
        w_down = [e for e in rec.events if e.kind == KEY and e.code == "w" and e.down]
        self.assertEqual(len(w_down), 30)  # one per hold, not ten
        self.assertTrue(any(e.code == "shift" for e in rec.events if e.kind == KEY))
        self.assertEqual(vk_name(0xBC), "comma")  # never a raw "," in the action text

    def test_raw_mouse_motion_buttons_and_wheel(self):
        rec = self.rec(1)
        moves = [e for e in rec.events if e.kind == MOVE]
        self.assertAlmostEqual(sum(e.dx for e in moves), 2 * 1000 * 10)  # 10 turning seconds at 1 kHz
        self.assertAlmostEqual(sum(e.dy for e in moves), -1 * 1000 * 10)
        self.assertLess(len(moves), 1200)  # binned to 10 ms, not 10k raw packets
        clicks = [e for e in rec.events if e.kind == BUTTON and e.code == "mouse_left"]
        self.assertEqual(sum(e.down for e in clicks), 30)
        self.assertEqual([e.dz for e in rec.events if e.kind == WHEEL], [-1.0])

    def test_frames_point_into_the_video(self):
        rec = self.rec(1)
        self.assertAlmostEqual(rec.meta.frame_hz, 60.0, delta=0.1)
        self.assertEqual(rec.observations[0].frame, "video.mkv#t=0.020")
        self.assertTrue((self.raw / rec.meta.recording_id / "video.mkv").is_symlink())

    def test_window_switches_are_focus_changes_that_the_gate_reads(self):
        rec = self.rec(0)
        focus = [(round(e.t, 1), e.code) for e in rec.events if e.kind == FOCUS]
        self.assertEqual(focus, [(10.9, "Discord"), (14.9, "game")])
        rows = {json.loads(line)["recording_id"]: json.loads(line)
                for line in (self.curated / "recordings.jsonl").read_text().splitlines()}
        alt = rows["d2e-minecraft-20250827_052328_366c9184_split_00"]
        self.assertGreater(alt["metrics"]["non_game_frac"], 0.1)
        self.assertFalse(alt["passed"])
        self.assertTrue(any(r.startswith("non_game_input") for r in alt["reasons"]))
        clean = rows["d2e-minecraft-20250827_052328_366c9184_split_01"]
        self.assertTrue(clean["passed"], clean["reasons"])
        self.assertEqual(clean["alignment"]["status"], "skipped")  # frames live in the video


class P2PSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rec = root / "p2p" / "0199-test"
        rec.mkdir(parents=True)
        frames = ["user"] * 200 + ["system"] * 40 + ["user"] * 180 + ["unknown"] * 10 + ["user"] * 30
        (rec / "annotation.proto").write_bytes(p2p_annotation(frames))
        (rec / "video.mp4").write_bytes(b"not really a video")
        cls.rows = run_ingest("p2p", root / "p2p", root / "raw", REGISTRY)
        cls.raw = root / "raw"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_policy_frames_split_the_file_into_human_runs(self):
        self.assertEqual([r["recording_id"] for r in self.rows], ["p2p-0199-test-00", "p2p-0199-test-01"])
        self.assertEqual([r["duration"] for r in self.rows], [10.0, 9.0])  # the 30-frame tail is too short
        self.assertEqual({r["session"] for r in self.rows}, {"p2p-0199-test"})
        self.assertEqual(self.rows[0]["stream"], "p2p:player7")
        self.assertEqual(self.rows[0]["world"], "roblox-blade-ball")
        self.assertEqual(human_runs(["user"] * 3 + ["system"] + ["user"] * 5, 4), ([(4, 9)], 1))

    def test_actions_keys_and_frames(self):
        rec = read_recording(self.raw / "p2p-0199-test-01")
        self.assertEqual(rec.meta.frame_hz, 20.0)
        self.assertEqual((rec.meta.width, rec.meta.height), (1920, 1080))
        # Second run starts at frame 240 of the source video.
        self.assertEqual(rec.observations[0].frame, "video.mp4#t=12.000")
        self.assertEqual(rec.observations[0].t, 0.0)
        moves = [e for e in rec.events if e.kind == MOVE]
        self.assertEqual({(e.dx, e.dy) for e in moves}, {(-7, 3)})  # negative int32 decoded
        codes = {e.code for e in rec.events if e.kind in (KEY, BUTTON)}
        self.assertEqual(codes, {"w", "shift", "mouse_left"})

    def test_instructions_ride_along_as_state(self):
        rec = read_recording(self.raw / "p2p-0199-test-00")
        self.assertEqual(rec.observations[3].state, {"instruction": "cross the bridge"})


class SessionSplitTest(unittest.TestCase):
    def test_parts_of_one_session_stay_together(self):
        wcfg = WindowConfig()
        rows, windows = [], {}
        for i in range(12):
            rid = f"g-{i}"
            rows.append({"recording_id": rid, "world": "g", "stream": "g", "duration": 40.0, "session": f"s{i // 2}"})
            windows[rid] = make_windows(rid, "g", "g", [False] * 200, wcfg)
        for i in range(3):
            rid = f"a-{i}"
            rows.append({"recording_id": rid, "world": "a", "stream": "a", "duration": 40.0})
            windows[rid] = make_windows(rid, "a", "a", [False] * 200, wcfg)
        for seed in range(8):
            s = split(rows, windows, REGISTRY, "a", [], {}, SplitConfig(clips_per_world=3), wcfg, seed=seed)
            held = {r for v in s.val_recordings.values() for r in v}
            for k in range(6):
                pair = {f"g-{2 * k}", f"g-{2 * k + 1}"}
                self.assertIn(len(pair & held), (0, 2), (seed, pair, held))


class RegistryCatalogTest(unittest.TestCase):
    def test_catalog_titles_resolve_and_dedupe(self):
        self.assertEqual(REGISTRY.label_for("Minecraft_1.21.8"), "minecraft")
        self.assertEqual(REGISTRY.label_for("Battlefield_6_Open_Beta"), "battlefield-6")
        self.assertEqual(REGISTRY.label_for("Brand New Game"), "brand-new-game")
        gta = REGISTRY.get("grand-theft-auto-v").overlap(REGISTRY.get("grand-theft-auto-san-andreas"))
        self.assertIn("franchise", gta)
        self.assertIn("franchise", REGISTRY.get("roblox-blade-ball").overlap(REGISTRY.get("roblox-tower-of-hell")))


def _real(env: str):
    path = os.environ.get(env, "")
    return unittest.skipUnless(path and Path(path).is_dir(), f"set {env} to a folder of real files")


class RealSamplesTest(unittest.TestCase):
    def _run(self, source, folder):
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_ingest(source, Path(folder), Path(tmp) / "raw", REGISTRY)
            ledger = curate(Path(tmp) / "raw", Path(tmp) / "curated", workers=1)
        self.assertTrue(rows)
        self.assertEqual(ledger["summary"]["certified"], len(rows), ledger["summary"])

    @_real("EMBODIGEN_D2E_SAMPLES")
    @unittest.skipUnless(HAS_ZSTD, "D2E files use zstd chunks")
    def test_real_d2e(self):
        self._run("d2e", os.environ["EMBODIGEN_D2E_SAMPLES"])

    @_real("EMBODIGEN_P2P_SAMPLES")
    def test_real_p2p(self):
        self._run("p2p", os.environ["EMBODIGEN_P2P_SAMPLES"])


if __name__ == "__main__":
    unittest.main()
