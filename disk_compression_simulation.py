####################################################################################################
# This file converts the intesity function of the disk to a 2D array of pixels.
# The simulation is based on the
# paper "Diametral compression test with composite disk for dentin bond strength measurement
# – Finite element analysis" by Shih-Hao Huanga, Lian-Shan Linb, Alex S.L. Fokb, Chun-Pin Lina.
####################################################################################################

import math
import numpy as np
import disk_compression_funcs as funcs
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

#########################
#### disk constants #####
#########################

# The radius of the disk
R = 3.25 * 0.01
# The thickness of the disk
T = 0.001
# Wave length of the light
l = 632 * 10 ** (-9)
# stress constant of the material
c = 7.8 * 10 ** (-11)

# The number of pixels in the x direction
N = 4000
# The number of pixels in the y direction
M = 4000

save_path = "/Users/noamcohen/Documents/Polo/red/test.png"

# the conversion from voltage to pressure, in newtons per volt.
volt_to_pressure = 1250

#######################################
##### define the disk ##################
#######################################


def in_disk(x, y, R):
    """Checks if the point (x, y) is in the disk.
    Args:
        x (int): x coordinate
        y (int): y coordinate
        N (int): number of pixels in the x direction
        M (int): number of pixels in the y direction

    Returns:
        bool: True if the point is in the disk, False otherwise.
    """
    # use the midpoint circle algorithm to check if the point is in the disk
    return x**2 + y**2 <= R**2


##########################
#### numpy simulation ####
##########################


def intensity_to_pixels(intensity: list[list[int]]) -> list[list[tuple[int]]]:
    """Converts the intensity array to values between 0 and 255.
    Args:
        intensity (list[list[int]]): intensity function of the disk

    Returns:
        list[list[int]]: 2D array of pixels.
    """
    # Create a 2D array of zeros
    pixels = np.zeros((N, M))

    # Find the maximum value in the intensity function
    max_intensity = np.amax(intensity)
    # Calculate the intensity function for each pixel
    for i in range(N):
        for j in range(M):
            pixels[i][j] = 256 - int((intensity[i][j] / max_intensity) * 256)
            x = -R + R * 2 * i / (N + 1)
            y = -R + R * 2 * j / (M + 1)
            if not in_disk(x, y, R):
                pixels[i][j] = 0
    return pixels


def circle_rotate(image, x, y, radius, degree):
    img_arr = np.asarray(image)
    box = (x - radius, y - radius, x + radius + 1, y + radius + 1)
    crop = image.crop(box=box)
    crop_arr = np.asarray(crop)
    # build the cirle mask
    mask = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(crop_arr.shape[0]):
        for j in range(crop_arr.shape[1]):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius**2:
                mask[i, j] = 1
    # create the new circular image
    sub_img_arr = np.empty(crop_arr.shape, dtype="uint8")
    sub_img_arr[:, :, :3] = crop_arr[:, :, :3]
    sub_img_arr[:, :, 3] = mask * 255
    sub_img = Image.fromarray(sub_img_arr, "RGBA").rotate(degree)
    i2 = image.copy()
    i2.paste(sub_img, box[:2], sub_img.convert("RGBA"))
    return i2


def get_disk_pixels(image) -> tuple:
    """get circle from image

    Args:
        image_name (str): image path

    Returns:
        tuple: (x, y, r) of the circle
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)
    # If some circle is found
    if circles is not None:
        # Get the (x, y, r) as integers
        circles = np.round(circles[0, :]).astype("int")

    return circles[0]


def main():
    points = np.zeros((N, M))
    # Recieve the pressure applied to the disk from the user
    p = (
        volt_to_pressure
        * (float(input("Enter the pressure applied to the disk: ")))
        + 17.24
    )
    # Calculate the intensity function for each pixel
    for i in range(M):
        for j in range(M):
            # only calculate the intensity function if the point is in the disk
            y = -R + R * 2 * i / (M + 1)
            x = -R + R * 2 * j / (M + 1)
            if in_disk(x, y, R):
                points[i][j] = funcs.intensity(x, y, R, T, p, l, c)
                # Convert the intensity function to a 2D array of pixels
            else:
                points[i][j] = 0
    pixels = intensity_to_pixels(points)

    # show image
    # print("Showing image...")
    # # show image as red
    plt.imshow(pixels)
    # plt.show()
    plt.axis("off")
    plt.imsave(save_path, pixels, cmap="inferno")
    plt.close()

    return


main()
