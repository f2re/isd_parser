"""
Reprojecting images from a Geostationary projection
---------------------------------------------------

This example demonstrates Cartopy's ability to project images into the desired
projection on-the-fly. The image itself is retrieved from a URL and is loaded
directly into memory without storing it intermediately into a file. It
represents pre-processed data from the Spinning Enhanced Visible and Infrared
Imager onboard Meteosat Second Generation, which has been put into an image in
the data's native Geostationary coordinate system - it is then projected by
cartopy into a global Miller map.

"""
# try:
#     from urllib2 import urlopen
# except ImportError:
#     from urllib.request import urlopen
from io import BytesIO

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import numpy as np

def geos_image():
    """
    Return a specific SEVIRI image by retrieving it from a github gist URL.

    Returns
    -------
    img : numpy array
        The pixels of the image in a numpy array.
    img_proj : cartopy CRS
        The rectangular coordinate system of the image.
    img_extent : tuple of floats
        The extent of the image ``(x0, y0, x1, y1)`` referenced in
        the ``img_proj`` coordinate system.
    origin : str
        The origin of the image to be passed through to matplotlib's imshow.

    """


    filename="20190921_MSG4.jpg"
    with open(filename, 'rb') as fin:
        img_handle = BytesIO(fin.read())
    img = plt.imread(img_handle, format="jpeg")
    img_proj = ccrs.Geostationary(satellite_height=31086000) #satellite_height=35786000
    # img_extent = (-371200, 371200, -371200, 371200)
    img_extent = (-5500000, 5500000, -5500000, 5500000)
    return img, img_proj, img_extent, 'upper'


def main():
    # 
    # coordinates of stantion
    # 
    lat,lon = 44.75, 1.4
    _extent = [lon-20,lon+20,lat-10,lat+10]
    # _extent = [lon-2,lon+2,lat-1,lat+1]

    ax = plt.axes(projection=ccrs.Miller())
    # ax = plt.axes(projection=ccrs.Orthographic(lat,lon))
    # ax = plt.axes(projection=ccrs.Miller())
    ax.coastlines()
    ax.set_global()
    # print(cartopy.config['data_dir'])
    img, crs, extent, origin = geos_image()

    # print(img)

    im=plt.imshow(img, transform=crs, extent=extent, origin=origin, cmap='gray')

    # ax.set_extent( _extent )
    ax.plot( lon,lat, 'bo', transform=ccrs.Geodetic()  )

    print(im)
    
    # print(im.shape)

    plt.show()


if __name__ == '__main__':
    main()
