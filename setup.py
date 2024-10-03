import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "CardioX"
AUTHOR_USER_NAME = "Udit Rawat"
SRC_REPO = "mlpro"
AUTHOR_EMAIL = "uditcsrawat@gmail.com"
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email="uditcsrawat@gmail.com",
    description="A small project pkg for end to end pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/udit-rawat/CardioX",
    project_urls={
        "Bug Tracker": f"https://github.com/udit-rawat/CardioX/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
