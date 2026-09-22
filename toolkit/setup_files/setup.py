"""
Setup package untuk proyek deteksi kendaraan
"""
from setuptools import setup, find_packages

setup(
    name="vehicle-detection",
    version="1.0.0",
    description="Sistem Deteksi dan Penghitungan Kendaraan",
    author="Tugas Akhir",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics",
        "opencv-python",
        "pyyaml",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "psutil",
    ],
)
