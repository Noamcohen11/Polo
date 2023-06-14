import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import argparse

#########################
### my test constants ###
#########################

######### image #########
IMAGE_FOLDER = "./images/"

# for each structure, (name, number of images, number of circles)
img_struct_dict = {
    "line": ("line", 8, 4),
    "square": ("square", 9, 9),
    "piramide": ("piramide", 8, 9),
    "rad": ("rad", 7, 6),
    "single": ("red", 11, 1),
}


def get_img_path(struct_name):
    (name, num_images, num_circles) = img_struct_dict[struct_name]
    image_path_list = [
        IMAGE_FOLDER + name + f"/{i}.png" for i in range(1, num_images + 1)
    ]
    return (image_path_list, num_circles)


######### videos #########
VIDEOS_FOLDER = "./videos/"

vid_struct_dict = {
    "line": ("line.mov", 4),
    "square": ("square.mov", 9),
    "single": ("single.mov", 1),
}


def get_vid_path(struct_name):
    (name, num_circles) = vid_struct_dict[struct_name]
    image_path = VIDEOS_FOLDER + name
    return (image_path, num_circles)


#####################################################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE
#####################################################################


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
            1.8,
            100,
            minRadius=70,
            maxRadius=200,
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
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord("q"):
            return True
    return False


def reorganize_disks(
    disks: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """Reorganize the disks by their y position, then x position.

    Args:
        disks (list[tuple[int, int, int, int]]): list of disks

    Returns: disks(list[tuple[int, int, int, int]]): reorganized list of disks
    """

    organized_disks = []
    if disks is None:
        return organized_disks
    for disk in disks:
        x, y, r, _ = disk
        if len(organized_disks) == 0:
            organized_disks.append(disk)
            continue

        for i in range(len(organized_disks)):
            x_org, y_org, r_org, _ = organized_disks[i]
            if y + r // 2 < y_org:
                organized_disks.insert(i, disk)
                break
            if y + r // 2 > y_org and y - r // 2 < y_org:
                if x < x_org:
                    organized_disks.insert(i, disk)
                    break
            if i == len(organized_disks) - 1:
                organized_disks.append(disk)
                break
    return organized_disks


def plot_all_disk_average_values(
    total_disks: list[list[tuple[int, int, int, int]]],
    plot_axis: tuple[int, int] = (3, 3),
) -> None:
    """Plot the average pixel values per disk.

    Args:
        disks (list[list[tuple[int, int, int, int]]]): list of disks
        plot_axis (tuple[int, int], optional): amount of plots per row and column. Defaults to (1,1).
    """
    if len(total_disks[0]) != plot_axis[0] * plot_axis[1]:
        raise ValueError(
            "The amount of disks does not match the amount of plots."
        )
    figure, axis = plt.subplots(plot_axis[0], plot_axis[1])
    image_index = list(range(len(total_disks)))
    image_index = [i * 2 for i in image_index]
    for i in range(len(total_disks[0])):
        avg_per_disk = []
        for image_disks in total_disks:
            avg_per_disk.append(image_disks[i][3])
        axis[i // plot_axis[0], i % plot_axis[0]].plot(
            image_index, avg_per_disk, marker=".", linestyle=""
        )
    plt.tight_layout()
    plt.show()


def plot_disk_average_values(
    total_disks: list[list[tuple[int, int, int, int]]]
) -> None:
    """Plot the average pixel values per disk.

    Args:
        disks (list[list[tuple[int, int, int, int]]]): list of disks
    """
    image_index = list(range(len(total_disks)))
    image_index = [i * 2 for i in image_index]
    for i in range(len(total_disks[0])):
        plt.figure(i)
        avg_per_disk = []
        for image_disks in total_disks:
            avg_per_disk.append(image_disks[i][3])
        plt.plot(image_index, avg_per_disk, marker=".", linestyle="")
        plt.show()


def disk_gradient(circle_pixels: np.ndarray) -> np.ndarray:
    i_minus_1 = np.roll(circle_pixels, 1, axis=0)
    i_plus_1 = np.roll(circle_pixels, -1, axis=0)
    j_minus_1 = np.roll(circle_pixels, 1, axis=1)
    j_plus_1 = np.roll(circle_pixels, -1, axis=1)
    i_minus_1_j_minus_1 = np.roll(np.roll(circle_pixels, 1, axis=0), 1, axis=1)
    i_plus_1_j_plus_1 = np.roll(np.roll(circle_pixels, -1, axis=0), -1, axis=1)
    i_plus_1_j_minus_1 = np.roll(np.roll(circle_pixels, -1, axis=0), 1, axis=1)
    i_minus_1_j_plus_1 = np.roll(np.roll(circle_pixels, 1, axis=0), -1, axis=1)

    term1 = 0.25 * (0.25 * (i_minus_1 - i_plus_1) ** 2)
    term2 = 0.125 * ((i_minus_1_j_minus_1 - i_plus_1_j_plus_1) ** 2)
    term3 = 0.25 * ((j_minus_1 - j_plus_1) ** 2)
    term4 = 0.125 * ((i_plus_1_j_minus_1 - i_minus_1_j_plus_1) ** 2)

    G = np.mean(0.25 * (term1 + term2 + term3 + term4))

    return G


def single_image_read(image, circle_num):
    circles = detect_circles(image, num_circles=circle_num)

    disks = []
    # Get the average pixel intensity of the disk.
    if len(circles) == 0:
        print("no circles found")
        return None

    # Get the average pixel intensity of the disk.
    for (x, y, r) in circles:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r // 10, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        circle_pixels = masked_image[mask > 0]
        circle_average = np.mean(circle_pixels)
        disks.append((x, y, r, circle_average))

    return disks


def images_disks(image_path_list, circle_num, debug_mode=False):
    """image_read function for the disk detection

    Args:
        image_path_list (list): list of image paths
        circle_num (int): number of circles to detect
    """

    total_disks = []
    for image_path in image_path_list:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Reorganize the disks by their x position, then y position.
        disks = reorganize_disks(single_image_read(image, circle_num))
        total_disks.append(disks)
        # Plot the image of the disks.
        if debug_mode:

            og_image = cv2.imread(image_path)
            stop = debug_show_disks(og_image, image, disks)
            if stop:
                break

    # Plot the average pixel values per disk.
    return total_disks


def vid_disks(video_path, circle_num, debug_mode=False):
    video = cv2.VideoCapture(video_path)
    total_disks = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disks = reorganize_disks(single_image_read(gray, circle_num))
        if len(disks) == circle_num:
            total_disks.append(disks)
        if debug_mode:
            stop = debug_show_disks(frame, gray, disks)
            if stop:
                video.release()
                break

    # clean up our resources

    return total_disks


if __name__ == "__main__":
    struct_type = "line"
    error = False

    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", help="Input string")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--vid", action="store_true", help="handle video input"
    )
    parser.add_argument(
        "--plot_all", action="store_true", help="plot all at one graph"
    )
    args = parser.parse_args()

    debug = False
    if args.debug:
        print("Debug mode is enabled!")
        debug = True

    # Choose plot function based on input
    struct_type = args.input_string
    if args.vid:
        struct_dict = vid_struct_dict
        plot_func = vid_disks
        get_path = get_vid_path
    else:
        struct_dict = img_struct_dict
        plot_func = images_disks
        get_path = get_img_path
    if struct_type not in struct_dict:
        error = True
    if error:
        print("Usage: python3 {} [struct_type]".format(sys.argv[0]))
        print("Available struct types: {}".format(struct_dict.keys()))
        sys.exit(1)

    (path_list, circle_num) = get_path(struct_type)
    disks = plot_func(path_list, circle_num, debug_mode=debug)
    if args.plot_all:
        plot_all_disk_average_values(disks, plot_axis=(3, 3))
    else:
        plot_disk_average_values(disks)
