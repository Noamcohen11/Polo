import cv2
import numpy as np
from disk_funcs import in_disk

# from disk_compression_simulation import in_disk

#########################
####### Constants #######
#########################

FOLDER_PATH = "/Users/noamcohen/Documents/Polo/"
DEBUG_MODE = True

########## LINE ##########
LINE_IMAGE_PATH_LIST = [FOLDER_PATH + f"line/{i}.png" for i in range(1, 8)]
LINE_CIRCLE_NUM = 4

########## SQUARE ##########
SQ_IMAGE_PATH_LIST = [FOLDER_PATH + f"square/{i}.png" for i in range(1, 9)]
SQ_CIRCLE_NUM = 9


#########################
######### funcs #########
#########################


def detect_circles(image, num_circles=4, max_iterations=10) -> list:
    """Detect a constant number of circles in an image.
       iterates multiple times to detect circles, in each iteration the image
       is rotated so that circles can be found.

    Args:
        image: The image to detect circles in
        num_circles (int, optional): The number of circles to detect. Defaults to 4.
        max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 10.

    Returns:
        list: list of circle centers. length of num_circles if possible.
    """
    # Apply a blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Initialize circle centers
    circle_centers = []

    best_res = 0

    # Iterate multiple times to detect circles
    loop_cnt = 0
    circle_parameter = 1.3
    for i in range(max_iterations):
        # Apply HoughCircles to detect circles
        # If we flipped back to the original image, change the parameters.
        if loop_cnt > 4:
            circle_parameter += (i - 4) * 0.3
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1.3,
            100,
            minRadius=70,
            # maxRadius=400,
        )

        # Ensure that circles are detected
        if circles is not None:
            # Convert the circle parameters to integers
            circles_np = np.round(circles[0, :]).astype("int")

            # Check if the desired number of circles is found
            if len(circles_np) >= num_circles:
                circle_centers = circles_np
                return circle_centers[:num_circles]

            if len(circles_np) > best_res:
                best_res = len(circles_np)
                circle_centers = circles_np
        # Rotate the image for the next iteration
        blurred = np.rot90(blurred)
        loop_cnt += 1

    if best_res > 0:
        return circle_centers[:best_res]
    return []


def debug_show_disks(og_image, image, disks):
    """Show the disks on the image

    Args:
        og_image (np.ndarray): original image
        image (np.ndarray): image with disks
        disks (list): list of disks
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x, y, r, disk_avg) in disks:
        cv2.circle(mask, (x, y), r, 255, -1)
        output = cv2.bitwise_and(image, image, mask=mask)

        left_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
        right_image = output

        # Ensure that both images have the same height
        desired_width = min(left_image.shape[1], right_image.shape[1])

        # Resize both images to have the desired width while maintaining the aspect ratio
        left_image_resized = cv2.resize(
            left_image,
            (
                desired_width,
                int(left_image.shape[0] * desired_width / left_image.shape[1]),
            ),
        )
        right_image_resized = cv2.resize(
            right_image,
            (
                desired_width,
                int(
                    right_image.shape[0] * desired_width / right_image.shape[1]
                ),
            ),
        )

        # Calculate the desired height based on the minimum height of both resized images
        desired_height = min(
            left_image_resized.shape[0], right_image_resized.shape[0]
        )

        # Resize both images to have the desired height
        left_image_resized = cv2.resize(
            left_image_resized, (desired_width, desired_height)
        )
        right_image_resized = cv2.resize(
            right_image_resized, (desired_width, desired_height)
        )

        # Concatenate the two images horizontally
        concatenated_image = np.concatenate(
            (left_image_resized, right_image_resized), axis=1
        )

        cv2.imshow("Circle", concatenated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return


def calculate_average_pixel_value(image, circles, threshold=40):
    # Create a mask to exclude the circular regions
    mask = np.zeros_like(image, dtype=np.uint8)
    for circle in circles:
        x, y, r = circle
        cv2.circle(mask, (x, y), r, (255), -1)

    # Apply the mask to exclude the circular regions from the image
    masked_image = cv2.bitwise_and(image, image, mask=~mask)

    # Calculate the average pixel value of the masked image
    masked_pixels = masked_image.flatten()
    filtered_pixels = masked_pixels[masked_pixels > threshold]
    if len(filtered_pixels) > 0:
        average_pixel_value = np.mean(filtered_pixels)
    else:
        average_pixel_value = 0

    return average_pixel_value


def main():
    for image_path in SQ_IMAGE_PATH_LIST:
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        circles = detect_circles(image, num_circles=SQ_CIRCLE_NUM)

        disks = []
        # Get the average pixel intensity of the disk
        if len(circles) == 0:
            print("no circles found")
            continue
        for (x, y, r) in circles:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r // 10, 255, -1)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            circle_pixels = masked_image[mask > 0]
            circle_average = np.mean(circle_pixels)
            disks.append((x, y, r, circle_average))

        if DEBUG_MODE:

            print(calculate_average_pixel_value(image, circles))
            print(disks)
            og_image = cv2.imread(image_path)
            debug_show_disks(og_image, image, disks)
    return


main()
