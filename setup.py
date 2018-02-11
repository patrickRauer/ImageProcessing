from setuptools import setup

setup(
    name='ImageProcessing',
    version='0.5',
    packages=['', 'ImageProcessing', 'ImageProcessing.reduction', 'ImageProcessing.astrometry',
              'ImageProcessing.extraction', 'ImageProcessing.photometry', 'ImageProcessing.calibration'],
    url='',
    license='GPL',
    author='Patrick Rauer',
    author_email='j.p.Rauer@sron.nl',
    description='Package for the automatic astrometry and photometry of astronomical images',
    install_requires = ['numpy', 'astropy', 'astroquery', 'sklearn', 'mpdaf', 'photutils', 'scipy', 'imageio']
)
