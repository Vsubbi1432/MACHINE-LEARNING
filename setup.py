from setuptools import setup, find_packages

setup(
    name="house-price-prediction",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26,<3",
        "pandas>=2.1",
        "scikit-learn>=1.4",
        "scipy>=1.11",
        "joblib>=1.3",
        "matplotlib>=3.8",
        "xgboost>=2.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.6",
        "python-multipart>=0.0.9",
        "rich>=13.7",
        "pytest>=8.0",
        "httpx>=0.27",
    ],
)