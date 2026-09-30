"""Entry point for ``python -m kerrray``; equivalent to the ``kerrray`` script."""

from kerrray.cli import app

if __name__ == "__main__":
    app(prog_name="kerrray")
