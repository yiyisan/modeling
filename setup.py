from setuptools import setup, find_packages

PACKAGE = "work"
NAME = "work"
DESCRIPTION = "rich help module for marvin modelling"
AUTHOR = "CreditX"
AUTHOR_EMAIL = "marvin@creditx.com"
URL = "https://git.creditx.com/marvin/marvin_modeling"
VERSION = '0.0.2'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="BSD",
    url=URL,
    include_package_data = True,
    packages=['work', 'work/marvin'],
    classifiers=[
        'License :: OSI Approved :: Private License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
)
