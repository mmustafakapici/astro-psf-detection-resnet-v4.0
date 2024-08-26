from setuptools import setup, find_packages

# Gereken bağımlılıkları tanımlayın
required_packages = [
    'torch==2.2.0',
    'torchvision==0.17.0',
    'numpy',
    'matplotlib',
    'astropy',
    'tqdm',
    'scikit-learn',
    'opencv-python-headless',
    'pandas',
    'pyyaml',
]

# Kurulum ayarları
setup(
    name='astro_psf_detection',
    version='0.1.0',
    description='A project for detecting and classifying astronomical sources in FITS images using deep learning.',
    author='Muhammed Mustafa KAPICI',
    author_email='m.mustafakapici@gmail.com',
    url='https://github.com/mmustafakapici/astro_psf_detection',  # Projenizin URL'si
    packages=find_packages(),  # Paketleri bul ve ekle
    install_requires=required_packages,  # Bağımlılıkları ekle
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    keywords='astronomy, PSF, deep learning, FITS, image processing',
    python_requires='>=3.7',  # Minimum Python sürümü
    license='MIT',
    include_package_data=True,  # Paket ile birlikte veri dosyalarını dahil et
    entry_points={
        'console_scripts': [
            'astro-psf-detect=src.train:main',
        ],
    },
)
