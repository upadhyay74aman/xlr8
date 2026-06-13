from setuptools import setup, find_packages

setup(
    name="xlr8",
    version="1.0.0",
    packages=find_packages(),
    py_modules=["main"],
    install_requires=[
        "fastapi",
        "uvicorn",
        "httpx",
        "huggingface_hub",
        "hf_transfer",
    ],
    entry_points={
        "console_scripts": [
            "xlr8=main:run_xlr8_sync",
        ],
    },
)