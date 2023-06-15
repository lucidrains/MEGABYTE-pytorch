from setuptools import setup, find_packages

setup(
  name = 'MEGABYTE-pytorch',
  packages = find_packages(),
  version = '0.2.0',
  license='MIT',
  description = 'MEGABYTE - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/MEGABYTE-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'torch>=1.10',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
