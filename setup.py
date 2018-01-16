import setuptools

setuptools.setup(
    name='neural_renderer',
    version='0.0.1',
    test_suite='tests',
    packages=['neural_renderer'],
    install_requires=['numpy', 'chainer', 'cupy', 'tqdm'],
)
