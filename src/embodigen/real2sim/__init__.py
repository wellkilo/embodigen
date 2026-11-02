"""Real -> sim -> real: a measured tabletop becomes a digital twin that generates certified data.

``twin.py`` holds the twin file and the scene generator that turns it (and randomised variants of
it) into the scenes the pipeline already plans, renders and validates. ``estimate.py`` asks a cloud
vision model for a first twin from one photo.
"""
