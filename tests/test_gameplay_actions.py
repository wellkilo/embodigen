import unittest

from embodigen.gameplay.actions import (
    KEYS_PER_SLOT,
    MOUSE_LIMIT,
    SUBSTEPS,
    Canonicalizer,
    canonical_key,
    decode_action,
    encode_action,
    turn_from_json,
    turn_to_json,
)
from embodigen.gameplay.types import BUTTON, KEY, MOVE, WHEEL, ActionChunk, InputEvent, Observation


def frames(seconds: float, hz: float = 30.0) -> list[Observation]:
    return [Observation(t=k / hz, index=k, frame=f"frames/{k:06d}.png") for k in range(int(seconds * hz))]


def press(t0: float, t1: float, code: str) -> list[InputEvent]:
    kind = BUTTON if code.startswith("mouse_") else KEY
    return [InputEvent(t=t0, kind=kind, code=code, down=True), InputEvent(t=t1, kind=kind, code=code, down=False)]


class ActionTextTest(unittest.TestCase):
    def test_round_trip(self):
        chunk = ActionChunk(dx=12, dy=-3, dz=1, keys=(("w",), ("w", "space"), (), ("w",), ("mouse_left",), ()))
        text = encode_action(chunk)
        self.assertEqual(
            text,
            "<|action_start|> 12 ; -3 ; 1 ; w ; w,space ; none ; w ; mouse_left ; none <|action_end|>",
        )
        self.assertEqual(decode_action(text), chunk)

    def test_decode_rejects_malformed_output(self):
        with self.assertRaises(ValueError):
            decode_action("<|action_start|> 1 ; 2 ; 3 <|action_end|>")
        with self.assertRaises(ValueError):
            decode_action("no action here")

    def test_decode_clips_out_of_range_values(self):
        chunk = decode_action("<|action_start|> 5000 ; -5000 ; 9 ; none ; none ; none ; none ; none ; none <|action_end|>")
        self.assertEqual((chunk.dx, chunk.dy, chunk.dz), (MOUSE_LIMIT, -MOUSE_LIMIT, 5))

    def test_canonical_key_names(self):
        self.assertEqual(canonical_key("key.keyboard.w"), "w")
        self.assertEqual(canonical_key("key.keyboard.left.shift"), "shift")
        self.assertEqual(canonical_key("key.keyboard.left.control"), "ctrl")
        self.assertEqual(canonical_key("Escape"), "esc")
        self.assertEqual(canonical_key("LMB"), "mouse_left")


class CanonicalizerTest(unittest.TestCase):
    def setUp(self):
        self.canon = Canonicalizer()

    def test_turns_are_5hz_with_six_slots(self):
        turns = self.canon.turns([], frames(2.0), duration=2.0)
        self.assertEqual(len(turns), 10)
        self.assertEqual([t.t for t in turns[:3]], [0.0, 0.2, 0.4])
        self.assertTrue(all(len(t.action.keys) == SUBSTEPS for t in turns))
        self.assertTrue(all(t.action.idle for t in turns))
        # The observation is the frame captured as the packet starts.
        self.assertEqual(turns[2].frame, "frames/000012.png")

    def test_mouse_is_summed_scaled_and_clipped(self):
        events = [InputEvent(t=0.01 * i, kind=MOVE, dx=10.0, dy=-2.0) for i in range(10)]  # all in turn 0
        events.append(InputEvent(t=0.25, kind=MOVE, dx=900.0))
        events.append(InputEvent(t=0.26, kind=MOVE, dx=900.0))
        events.append(InputEvent(t=0.45, kind=WHEEL, dz=1.0))
        turns = self.canon.turns(events, frames(1.0), duration=1.0, mouse_scale=0.5)
        self.assertEqual((turns[0].action.dx, turns[0].action.dy), (50, -10))
        self.assertEqual(turns[1].action.dx, 900)  # (900 + 900) * 0.5
        self.assertEqual(turns[2].action.dz, 1)
        turns = self.canon.turns(events, frames(1.0), duration=1.0, mouse_scale=1.0)
        self.assertEqual(turns[1].action.dx, MOUSE_LIMIT)

    def test_key_slots_follow_33ms_substeps(self):
        # Hold w across the first half of turn 0 only.
        events = press(0.0, 0.1, "w")
        turn = self.canon.turns(events, frames(0.4), duration=0.4)[0]
        self.assertEqual(turn.action.keys[:3], (("w",), ("w",), ("w",)))
        self.assertEqual(turn.action.keys[3:], ((), (), ()))

    def test_short_tap_is_not_lost(self):
        events = press(0.105, 0.11, "space") + press(0.15, 0.15, "e")  # 5 ms tap, zero-length tap
        turn = self.canon.turns(events, frames(0.4), duration=0.4)[0]
        held = [k for slot in turn.action.keys for k in slot]
        self.assertIn("space", held)
        self.assertIn("e", held)

    def test_slot_overflow_keeps_movement_first(self):
        events = []
        for code in ("1", "e", "w", "mouse_left", "space", "d"):
            events += press(0.0, 0.2, code)
        turn = self.canon.turns(events, frames(0.4), duration=0.4)[0]
        self.assertEqual(turn.action.keys[0], ("w", "d", "mouse_left", "space"))
        self.assertEqual(len(turn.action.keys[0]), KEYS_PER_SLOT)
        self.assertEqual(turn.truncated_keys, 2 * SUBSTEPS)

    def test_key_still_down_at_the_end_counts_as_held(self):
        events = [InputEvent(t=0.3, kind=KEY, code="w", down=True)]
        turns = self.canon.turns(events, frames(0.6), duration=0.6)
        self.assertEqual(turns[2].action.keys, (("w",),) * SUBSTEPS)

    def test_turn_json_round_trip(self):
        events = press(0.0, 0.3, "w") + [InputEvent(t=0.05, kind=MOVE, dx=7.0)]
        turn = self.canon.turns(events, frames(0.4), duration=0.4)[0]
        self.assertEqual(turn_from_json(turn_to_json(turn)).action, turn.action)


if __name__ == "__main__":
    unittest.main()
