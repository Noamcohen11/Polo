####################################################################################################
# This file converts the intesity function of the disk to a 2D array of pixels.
# The simulation is based on the
# paper "Diametral compression test with composite disk for dentin bond strength measurement
# – Finite element analysis" by Shih-Hao Huanga, Lian-Shan Linb, Alex S.L. Fokb, Chun-Pin Lina.
####################################################################################################

import numpy as np
import disk_compression_funcs as stress_funcs
import matplotlib.pyplot as plt
from disk_funcs import in_disk

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
                points[i][j] = stress_funcs.intensity(x, y, R, T, p, l, c)
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
