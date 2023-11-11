from setuptools import setup

setup(
     # Needed to silence warnings (and to be a worthwhile package)
    name='public_wifi',
    url='https://github.com/aggle/public_wifi',
    author='Jonathan Aguilar',
    author_email='jaguilar@stsci.edu',
    # Needed to actually package something
    packages=['public_wifi'],
    # Needed for dependencies
    install_requires=['matplotlib',
                      'numpy',
                      'pandas',
                      'reproject',
                      'astropy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Software for performing PSF subtraction on sources in wide-field mosaics',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.rst').read(),   
)
