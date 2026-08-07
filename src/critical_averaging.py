"""Compatibility entry point for the v1 prototype command.

The implemented CLI now lives at ``python -m tna.cli``. This wrapper keeps
older notes, scripts, and teaching material from breaking while still routing
all execution through the modular package.
"""

from tna.cli import main


if __name__ == "__main__":
    main()
