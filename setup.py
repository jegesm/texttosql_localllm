from setuptools import setup, find_packages

setup(
    name="texttosql_localllm",          # Replace with your package name
    version="0.1.0",            # Start with a simple version
    packages=find_packages(),   # Automatically find packages in your directory
    install_requires=[          # List your package dependencies here
        # "some_dependency>=1.0",
    ],
    author="David Visontai",
    author_email="david.visontai@ttk.elte.hu",
    description="Run text to sql workflows and benchmarks using Ollama and Langchain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jegesm/texttosql_localllm.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',    # Specify the Python versions you support
)

