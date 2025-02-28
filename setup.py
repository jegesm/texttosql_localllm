from setuptools import setup, find_packages

setup(
    name="texttosql_localllm",          # Replace with your package name
    version="0.1.3",            # Start with a simple version
    packages=find_packages(),   # Automatically find packages in your directory
    install_requires=[          # List your package dependencies here
        'chromadb==0.6.3',
        'langchain==0.3.19',
        'langchain-chroma==0.2.2',
        'langchain-community==0.3.18',
        'langchain-core==0.3.39',
        'langchain-experimental==0.3.4',
        'langchain-ollama==0.2.3',
        'langchain-openai==0.3.7',
        'langchain-text-splitters==0.3.6',
        'sql_metadata',
        'sqlalchemy',
        'sqlvalidator'
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
    python_requires='>=3.9',    # Specify the Python versions you support
)

