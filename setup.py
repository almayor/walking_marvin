from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='walking_marvin',
    version='0.1.0',
    install_requires=requirements,
    url='github.com/almayor/walking_marvin',
    license='MIT',
    author='almayor',
    author_email='mayorovme@gmail.com',
    description='Teach Marvin to walk using OpenAI Gym and neuroevolution'
)
