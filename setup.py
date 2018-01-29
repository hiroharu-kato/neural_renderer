import setuptools

from neural_renderer import __version__

setuptools.setup(
    version=__version__,
    name='neural_renderer',
    test_suite='tests',
    packages=['neural_renderer'],
    install_requires=['numpy', 'chainer', 'cupy', 'tqdm'],
)
