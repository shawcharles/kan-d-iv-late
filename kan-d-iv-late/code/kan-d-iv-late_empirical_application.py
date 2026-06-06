"""Compatibility shim for the packaged empirical module."""

from kan_d_iv_late.empirical import *  # noqa: F401,F403
from kan_d_iv_late.empirical import main


if __name__ == "__main__":
    main()
