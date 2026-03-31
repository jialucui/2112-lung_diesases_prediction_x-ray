from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pneumonia-detection',
    version='0.1.0',
    author='jialucui',
    author_email='jcui0208@uni.sydney.edu.au',
    description='Medical image analysis for pneumonia detection from chest X-rays using deep learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jialucui/2112-Enviromental-prediction',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pandas>=1.5.0',
        'scikit-learn>=1.2.0',
        'opencv-python>=4.7.0',
        'PyYAML>=6.0',
    ],
)
