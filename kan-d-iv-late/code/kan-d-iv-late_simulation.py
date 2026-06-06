"""Compatibility shim for the packaged simulation module."""

from kan_d_iv_late.simulation import *  # noqa: F401,F403
from kan_d_iv_late.simulation import main


if __name__ == "__main__":
    main()
