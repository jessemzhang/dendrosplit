from distutils.core import setup
setup(
  name = 'dendrosplit',
  packages = ['dendrosplit'], # this must be the same as the name above
  version = '0.1',
  description = 'A Python 2.7 package for performing interpretable clustering and feature selection for single-cell RNA-Seq datasets.',
  author = 'Jesse M. Zhang, Jue Fan, H. Christina Fan, David Rosenfeld, David Tse',
  author_email = 'jessez@stanford.edu',
  url = 'https://github.com/jessemzhang/dendrosplit', # use the URL to the github repo
  download_url = 'https://github.com/jessemzhang/dendrosplit/archive/0.1.tar.gz',
  keywords = ['single-cell', 'rna-seq', 'clustering'], # arbitrary keywords
  classifiers = [],
)
