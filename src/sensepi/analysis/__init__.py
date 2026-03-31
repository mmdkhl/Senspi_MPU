"""Signal analysis utilities (FFT, filtering, and feature extraction).

This package gathers pure-Python helpers that operate on NumPy arrays of
sensor samples. Modules such as :mod:`fft`, :mod:`filters`, :mod:`features`,
and :mod:`rate` stay free of Qt and I/O dependencies so they can be reused in
command-line scripts, automated tests, or GUI tabs alike.
"""
