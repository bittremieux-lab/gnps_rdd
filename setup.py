import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gfop-ka-west",
    version="0.0.1",
    author="Kiana West",
    author_email="kiana.a.west@gmail.com",
    description="Global FoodOmics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ka-west/gfop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={"": ["data/*.txt"]},
)
