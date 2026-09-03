"""Setup script for Vehicle Detection System"""

from setuptools import setup, find_packages

setup(
    name="vehicle-detection-itera",
    version="1.0.0",
    author="Penelitian Tugas Akhir",
    description="Sistem Perhitungan Jumlah Kendaraan Bermotor Berbasis Pengolahan Citra - UPT K3L ITERA",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "vehicle-train=src.train:main",
            "vehicle-detect=src.detect:main",
            "vehicle-eval=src.evaluate:main",
            "vehicle-realtime=src.realtime:main",
        ],
    },
)
