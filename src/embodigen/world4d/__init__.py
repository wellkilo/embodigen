"""4D worlds: the pipeline's exact geometry as 3D Gaussian splats over time.

``robot.episode_world`` samples a robot episode's scene; ``game.recording_world``
lifts an engine rollout from its depth and pose; ``world.write_world`` stores
either for the web viewer (``embodigen ui`` -> 4D 世界) and for 3DGS tools.
"""
