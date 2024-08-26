import os
from concurrent.futures import ThreadPoolExecutor
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle
from tqdm import tqdm
from utils import load_config

def download_fits_for_set(set_index, ra, dec, radius=0.02, width=512, base_dir='trainingset', bands=['g', 'r', 'z']):
    set_dir = os.path.join(base_dir, f'set_{set_index}')
    os.makedirs(set_dir, exist_ok=True)

    for band in bands:
        pos = coords.SkyCoord(ra, dec, unit="deg", frame="icrs")
        radius = Angle(radius, unit='deg')
        images = SDSS.get_images(coordinates=pos, band=band, radius=radius)

        if images:
            image_data = images[0][0].data
            wcs = WCS(images[0][0].header)
            cutout = cutout_image(image_data, wcs, width)

            fits_filename = os.path.join(set_dir, f'img_{band}.fits')
            hdu = fits.PrimaryHDU(data=cutout, header=wcs.to_header())
            hdu.writeto(fits_filename, overwrite=True)
        else:
            print(f"No image found for {band}-band at RA: {ra}, DEC: {dec}")

def cutout_image(image_data, wcs, width):
    center_x, center_y = np.array(image_data.shape) // 2
    x_start = center_x - width // 2
    y_start = center_y - width // 2
    cutout = image_data[y_start:y_start + width, x_start:x_start + width]
    return cutout

def main():
    # Konfigürasyon dosyasını yükle
    config = load_config()

    num_train = config['data']['train_fits_number']
    num_val = config['data']['val_fits_number']
    sets_to_download = num_train + num_val

    # Örnek RA ve DEC değerleri (bunları rastgele seçebilir ya da SDSS'den alabilirsiniz)
    ra_dec_list = [(180.0 + i*0.1, 30.0 + i*0.1) for i in range(sets_to_download)]

    training_set_dir = config['data']['training_set_dir']
    validation_set_dir = config['data']['validation_set_dir']
    bands = config['data']['bands']

    def process_set(i, ra_dec):
        ra, dec = ra_dec
        if i < num_train:
            download_fits_for_set(i, ra, dec, base_dir=training_set_dir, bands=bands)
        else:
            val_set_index = i - num_train  # Validation setler için index 0'dan başlar
            download_fits_for_set(val_set_index, ra, dec, base_dir=validation_set_dir, bands=bands)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(process_set, range(sets_to_download), ra_dec_list), total=sets_to_download))

if __name__ == "__main__":
    main()
