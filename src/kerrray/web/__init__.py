"""KerrRay Desktop: a local browser front end for the KerrRay engine.

``kerrray gui`` starts :func:`kerrray.web.server.serve`, a standard-library
HTTP server that serves the static desktop UI from ``static/`` and a small
JSON API (:mod:`kerrray.web.api`). Every number the UI shows is computed by
the engine on request; nothing is precomputed or hard-coded.
"""
