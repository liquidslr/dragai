from setuptools import setup, find_packages

version = "0.0.1"
DESCRIPTION = "🚀🤖 dragai: Open-source RAG agent for websites"
LONG_DESCRIPTION = open("README.md", encoding="utf-8").read()

setup(
    name="dragai",
    version=version,
    author="Gaurav Kumar",
    author_email="gaurav,kumar9825@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license="MIT",
    packages=find_packages(),
    install_requires=[],
    keywords=["python", "rag agent", "ai"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
)
