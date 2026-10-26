"""Virtual-world track: gameplay data for embodied agents.

Games are the simulation environments. A sample is what the player saw and
the keyboard-mouse action they took, packaged the way GameScaling trains on
it: 5 Hz observations, 200 ms action packets of six 33 ms key slots, 20-turn
windows, mixtures over many worlds with nested data budgets, and three
levels of validation.

Pipeline: ``rollout``/``ingest`` -> ``curate`` -> ``build``; see
:mod:`embodigen.gameplay.pipeline`.
"""
