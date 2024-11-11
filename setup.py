from setuptools import setup, find_packages

setup(
    name="llm_mapreduce",
    version="0.1.0",
    description="MapReduce-inspired framework for extending context windows in large language models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Joseph Oladokun",
    author_email="oladokunjoseph2@example.com",
    url="https://github.com/Godskid89/LLM_MapReduce",
    py_modules=["mapreduce", "confidence_calibration", "structured_info_protocol", "utils"],
    packages=find_packages(),
    install_requires=[
        "tqdm>=4.62.0",
        "pytest>=7.0.0"
    ],
    extras_require={
        "openai": ["openai>=0.27.0"],
        "huggingface": ["transformers>=4.0.0"],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
