"""
run.py  —  single-file launcher for the OpenSees Model Updating package.

Usage (from this folder, with the senspi_mpu conda environment active):

    conda activate senspi_mpu
    python run.py
"""

from opensees_model_updating.cli.main import main

if __name__ == "__main__":
    main()
