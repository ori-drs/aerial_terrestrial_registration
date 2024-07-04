from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["digiforest_registration"], package_dir={"": "src"}
)

setup(**d)


# from distutils.core import setup

# setup(
#     name="digiforest_registration",
#     version="0.0.0",
#     author="Benoit Casseau",
#     author_email="benoit@robots.ox.ac.uk",
#     packages=["digiforest_registration"],
#     package_dir={"": "src"},
#     python_requires=">=3.6",
#     description="Python tools for DigiForest",
# )
