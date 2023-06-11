import numpy as np
from PIL import Image


def in_disk(x, y, R, center_x=0, center_y=0):
    """Checks if the point (x, y) is in the disk.
    Args:
        x (int): x coordinate
        y (int): y coordinate
        center_x (int): x coordinate of the center of the disk
        center_y (int): y coordinate of the center of the disk

    Returns:
        bool: True if the point is in the disk, False otherwise.
    """
    # use the midpoint circle algorithm to check if the point is in the disk
    return (center_x - x) ** 2 + (center_y - y) ** 2 <= R**2


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
