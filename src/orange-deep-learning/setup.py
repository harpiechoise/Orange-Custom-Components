from setuptools import setup

setup(
    name="DeepLearning",
    packages=['orangedeeplearning'],
    package_data={'orangedeeplearning': ['icons/*.svg']},
    classifiers=['Example :: Invalid'],
    entry_points={'orange.widgets': 'DeepLearning = orangedeeplearning'}
)
