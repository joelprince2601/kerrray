"""Camera, observer image plane, ray initialisation, renderer and shadow
reconstruction (PROJECT.md sections 15 to 17).

Phase 5 (Ray Tracing). ``camera.py`` (camera role) provides the ZAMO
observer, the pixel to celestial-coordinate mapping and the null,
backward-traced initial states (``Camera``, ``zamo_tetrad``,
``initial_states``; derivation in ``docs/raytracing.md``). ``rays.py``,
``shadow.py``, ``boundary.py``, ``renderer.py`` and ``backends/`` belong to
the raytracing role. Nothing is re-exported here yet; import from the
submodules.
"""
