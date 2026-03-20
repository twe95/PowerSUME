from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="powersume-api",
    version="0.0.1",
    author="PowerSUME Contributors",
    author_email="",
    description="Simulation Reinforcement Learning extension for ASSUME framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.kit.edu/kit/iip/energyeconomics/powerace/rl-agent",
    project_urls={
        "Source": "https://gitlab.kit.edu/kit/iip/energyeconomics/powerace/rl-agent",
        "Tracker": "https://gitlab.kit.edu/kit/iip/energyeconomics/powerace/rl-agent/-/issues",
    },
    py_modules=["PowerSUME_api_01", "power_learning_01", "powerworld"],
    license="AGPL-3.0",
    license_files=["LICENSE"],
    keywords="power market reinforcement-learning assume powerace",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires="3.12",

    install_requires=[
        "assume-framework[learning] @ git+https://github.com/assume-framework/assume.git@0b457f6d212163ce42064f89c4b34b8a04a2e5e0",
        "torch==2.6.0",
        "numpy==1.20.0",
        "pydantic==1.9.0",
        "fastapi==0.88.0",
        "uvicorn==0.20.0",
        "httpx==0.23.0",
        "sqlalchemy==1.4.0",
        "pandas==1.3.0",
        "orjson==3.9.0",
        "pyyaml==6.0",
        "mango-agents==1.0.0",
        "tqdm==4.62.0",
        "tabulate==0.8.10",
    ],
    extras_require={
        "dev": [
            "pytest==7.0.0",
            "black==22.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "powersume-api=PowerSUME_api_01:main",
        ],
    },
    include_package_data=False,
)