
from photutils import detect_sources, deblend_sources, DAOStarFinder
from photutils.segmentation import SourceCatalog
from astropy.stats import mad_std
from skimage.filters import gaussian
import numpy as np
import sep
import cv2




def detect_stars(data: np.ndarray) -> list:
    """
    Detects stars in a FITS image and returns a list of bounding boxes for each star.

    Parameters:
    -----------
    data : np.ndarray
        2D numpy array (image data) where stars will be detected.

    Returns:
    -------
    bounding_boxes : list
        A list containing bounding boxes for each detected star.
        Each box is defined by coordinates in the format (x1, y1, x2, y2).
    """

    # Compute the median absolute deviation of the image data, representing the noise level.
    data_std = mad_std(data)

    # DAOStarFinder is used to detect stars.
    daofind = DAOStarFinder(fwhm=2.5, threshold=3. * data_std)

    # Perform star detection.
    sources = daofind(data)

    bounding_boxes = []  # List to store bounding boxes.

    if sources is not None:
        for source in sources:
            x, y = int(source['xcentroid']), int(source['ycentroid'])

            # Adjust the size of the bounding box based on the FWHM or sharpness.
            size = int(source['fwhm'] * 3) if 'fwhm' in sources.colnames else int(source['sharpness'] * 8)

            x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
            x2, y2 = min(data.shape[1], x + size // 2), min(data.shape[0], y + size // 2)

            bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes


def detect_galaxies(data: np.ndarray) -> list:
    """
    Detects galaxies in a FITS image and returns a list of bounding boxes for each galaxy.

    Parameters:
    -----------
    data : np.ndarray
        2D numpy array (image data) where galaxies will be detected.

    Returns:
    -------
    bounding_boxes : list
        A list containing bounding boxes for each detected galaxy.
        Each box is defined by coordinates in the format (x1, y1, x2, y2).
    """

    # Apply Gaussian smoothing with a finer sigma value.
    smooth_data = gaussian(data, sigma=1.5)

    # Compute a more precise threshold value.
    threshold = 1.5 * np.std(smooth_data)

    # Detect sources using a lower npixels value for finer detection.
    segm = detect_sources(smooth_data, threshold, npixels=20)

    # Deblend the sources for more accurate segmentation.
    segm_deblend = deblend_sources(smooth_data, segm, npixels=5, nlevels=64, contrast=0.0001)

    # Create a source catalog for detected sources.
    cat = SourceCatalog(smooth_data, segm_deblend)

    bounding_boxes = []  # List to store bounding boxes.

    for source in cat:
        x, y = int(source.xcentroid), int(source.ycentroid)

        # Adjust the size of the bounding box based on the semimajor axis.
        size = int(source.semimajor_sigma.value * 4)

        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(data.shape[1], x + size // 2), min(data.shape[0], y + size // 2)

        bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes


def detect_lines(data: np.ndarray) -> list:
    """
    Detects lines in a FITS image and returns a list of bounding boxes for each line.

    Parameters:
    -----------
    data : np.ndarray
        2D numpy array (image data) where lines will be detected.

    Returns:
    -------
    bounding_boxes : list
        A list containing bounding boxes for each detected line.
        Each box is defined by coordinates in the format (x1, y1, x2, y2).
    """

    # Apply Gaussian smoothing to reduce noise before edge detection.
    smooth_data = gaussian(data, sigma=1.0)

    # Perform edge detection using Canny with adjusted parameters.
    edges = cv2.Canny(smooth_data.astype(np.uint8), 30, 100, apertureSize=3)

    # Detect lines using the Hough Line Transform.
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    bounding_boxes = []  # List to store bounding boxes.

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))

            # Calculate the bounding box for the detected line.
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes
""""
def detect_sources_sep(data: np.ndarray , threshold_sigma: float = 3.0, min_area: int = 5) -> list:
    
    
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms

   

    # Perform source detection using SEP.
    data_std = np.std(data)
    sources = sep.extract(data, threshold_sigma * data_std, minarea=min_area)
   
    #print found sources len
    print(f"Found {len(sources)} sources" + " with threshold_sigma: " + str(threshold_sigma) + " and min_area: " + str(min_area)  )
    
    bounding_boxes = []  # List to store bounding boxes.

    if sources is not None:
        for source in sources:
            x, y = int(source['x']), int(source['y'])

            # Adjust the size of the bounding box based on the elliptical radius.
            size = int(source['a'] * 4)

            x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
            x2, y2 = min(data.shape[1], x + size // 2), min(data.shape[0], y + size // 2)

            bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes
"""


def detect_sources_sep(data: np.ndarray, threshold_sigma: float = 3.0, min_area: int = 5,
                       filter_kernel=None, filter_type="matched", 
                       deblend_nthresh: int = 32,deblend_cont: float = 0.005, 
                       clean: bool = True, clean_param: float = 1.0,
                       gain: float = None, subpix: int = 0) -> list:
    """
    Detects sources in a FITS image using SEP and returns a list of bounding boxes for each source.

    Parameters:
    -----------
    data : np.ndarray
        2D numpy array (image data) where sources will be detected.
    threshold_sigma : float
        Detection threshold in terms of the standard deviation of the data.
    min_area : int
        Minimum area (in pixels) for detected sources.
    filter_kernel : array-like, optional
        Convolution kernel for filtering the image before detection.
    filter_type : str, optional
        Type of filtering to apply ('matched', 'convolution', 'none').
    deblend_nthresh : int, optional
        Number of thresholds used for deblending.
    deblend_cont : float, optional
        Minimum contrast ratio used for deblending sources.
    clean : bool, optional
        Whether to clean the source list by removing spurious detections.
    clean_param : float, optional
        Parameter controlling the aggressiveness of cleaning.
    gain : float, optional
        Gain for the image data (e.g., ADU/electron).
    subpix : int, optional
        Subpixel accuracy for source position determination.
        
    Returns:
    -------
    bounding_boxes : list
        A list containing bounding boxes for each detected source.
        Each box is defined by coordinates in the format (x1, y1, x2, y2).
    """

    data_std = np.std(data)

    # Perform source detection using SEP.
    sources = sep.extract(data, threshold_sigma * data_std,
                          minarea=min_area,
                          filter_kernel=filter_kernel,
                          filter_type=filter_type,
                          deblend_nthresh=deblend_nthresh,
                          deblend_cont=deblend_cont, 
                          clean=clean,
                          clean_param=clean_param,
                          gain=gain,
                          #subpix=subpix

                          )
    
    

    # Print the number of found sources.
    print(f"Found {len(sources)} sources in the image.")

    bounding_boxes = []  # List to store bounding boxes.

    if sources is not None:
        for source in sources:
            x, y = int(source['x']), int(source['y'])

            # Adjust the size of the bounding box based on the elliptical radius.
            size = int(source['a'] * 4)

            x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
            x2, y2 = min(data.shape[1], x + size // 2), min(data.shape[0], y + size // 2)

            bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes
