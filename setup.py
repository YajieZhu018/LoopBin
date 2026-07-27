from setuptools import setup
setup(
    name="loopbin",
    version="0.1.0",
    description="VaDE-based clustering of chromatin loops from Micro-C + CUT&Tag",
    python_requires=">=3.7",
    packages=["loopbin", "loopbin.fn", "loopbin.model", "loopbin.plot"],
    install_requires=[
        "tensorflow==2.5.0", "numpy", "pandas", "scipy", "scikit-learn",
        "scikit-image", "matplotlib", "seaborn", "cooler", "kneed", "tqdm", "biopython",
    ],
    entry_points={"console_scripts": ["loopbin = loopbin.cli:cli"]},
)
