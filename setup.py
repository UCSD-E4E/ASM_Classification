from setuptools import setup, find_packages
import ASMClassification
setup(
    name='ASMClassification',
    version=ASMClassification.__VERSION__,
    author='UCSD Engineers for Exploration',
    author_email='e4e@eng.ucsd.edu',
    entry_points={
        'console_scripts': [
            'ASMInference = ASMClassification.inference_main:main'
        ]
    },
    packages=find_packages(),
    install_requires=[
        'watchdog',
        'numpy',
        'torch',
        'tqdm',
        'sklearn',
        'pandas',
        'torchvision',
        'opencv-python',
        "PyGithub",
    ]
)