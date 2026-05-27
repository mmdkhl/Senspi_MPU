"""
run.py  —  single-file launcher for the OpenSees Model Updating package.

Usage (from the project root, with the 'opensees' conda environment active):

    conda activate opensees
    python run.py
"""

from src.cli.main import main

if __name__ == "__main__":
    main()
