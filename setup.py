from setuptools import setup, find_packages

setup(
    name='algo_trading_bot',
    version='0.1.0',
    description='A simple algorithmic trading bot',
    author='Ido Golberg',
    license='MIT',

    author_email='ido.goldb@gmail.com',
    packages=find_packages(include=['src', 'src.*', 'model', 'data', 'data.*', 'model.*']),
    install_requires=[
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'run-trading-bot=scripts.run_trading_bot:main',
        ],
    },
    test_suite='tests',
)
