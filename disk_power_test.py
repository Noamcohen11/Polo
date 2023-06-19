import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import argparse
import csv

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
    "line": ("line 2.mov", 4),
    "square": ("square2.mov", 9),
    "single": ("single 2.5.mov", 1),
    "pir": ("pir.mov", 9),
}


def get_vid_path(struct_name):
    (name, num_circles) = vid_struct_dict[struct_name]
    image_path = VIDEOS_FOLDER + name
    return (image_path, num_circles)


CSV_FOLDER = "./csv/"

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
            minRadius=30,
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

        gradient_magnitude = disk_gradient(image)

        output = cv2.bitwise_and(
            gradient_magnitude, gradient_magnitude, mask=mask
        )

        # output = cv2.bitwise_and(image, image, mask=mask)

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

        left_image_resized = cv2.applyColorMap(
            left_image_resized, cv2.COLORMAP_INFERNO
        )
        right_image_resized = cv2.normalize(
            right_image_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        right_image_resized = cv2.cvtColor(
            right_image_resized, cv2.COLOR_GRAY2BGR
        )
        right_image_resized = cv2.applyColorMap(
            right_image_resized, cv2.COLORMAP_INFERNO
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
    csv_file = CSV_FOLDER + "single 2.5.csv"

    stress = get_fitting_stress(
        30, [image[0][3] for image in total_disks], csv_file
    )

    for i in range(len(total_disks[0])):
        plt.figure(i)
        avg_per_disk = []
        for image_disks in total_disks:
            avg_per_disk.append(image_disks[i][3])
        plt.plot(image_index, avg_per_disk, marker=".", linestyle="")
        plt.show()


def disk_gradient(image) -> np.ndarray:
    # Pad the image to handle border pixels
    padded_image = np.pad(image, 1, mode="edge")

    # Extract the required pixels
    i_minus_1 = padded_image[:-2, 1:-1]
    i_plus_1 = padded_image[2:, 1:-1]
    j_minus_1 = padded_image[1:-1, :-2]
    j_plus_1 = padded_image[1:-1, 2:]
    i_minus_1_j_minus_1 = padded_image[:-2, :-2]
    i_plus_1_j_plus_1 = padded_image[2:, 2:]
    i_plus_1_j_minus_1 = padded_image[2:, :-2]
    i_minus_1_j_plus_1 = padded_image[:-2, 2:]

    # Calculate the gradient terms
    term1 = 0.25 * (0.25 * (i_minus_1 - i_plus_1) ** 2)
    term2 = 0.125 * ((i_minus_1_j_minus_1 - i_plus_1_j_plus_1) ** 2)
    term3 = 0.25 * ((j_minus_1 - j_plus_1) ** 2)
    term4 = 0.125 * ((i_plus_1_j_minus_1 - i_minus_1_j_plus_1) ** 2)

    # Calculate the mean gradient
    gradient = np.mean(0.25 * (term1 + term2 + term3 + term4))

    # Calculate the gradient image
    gradient_image = np.sqrt(term1 + term2 + term3 + term4)

    return gradient_image


def single_image_read(image, circle_num, circles=None):
    if circles is None:
        circles = detect_circles(image, num_circles=circle_num)

    disks = []
    # Get the average pixel intensity of the disk.
    if len(circles) == 0:
        print("no circles found")
        return None

    # Get the average pixel intensity of the disk.
    for (x, y, r) in circles:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
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
    total_gradient = []
    disks = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gradient_magnitude = disk_gradient(gray)
        if disks is None:
            disks = reorganize_disks(single_image_read(gray, circle_num))
            circles = [tuple[:3] for tuple in disks]
        else:
            disks = single_image_read(gray, circle_num, circles=circles)

        gradient_disks = single_image_read(
            gradient_magnitude, circle_num, circles=circles
        )
        if len(disks) >= circle_num:
            total_disks.append(disks)
            total_gradient.append(gradient_disks)
        if debug_mode:
            stop = debug_show_disks(frame, gray, disks)
            if stop:
                video.release()
                break

    # clean up our resources

    return (total_disks, total_gradient)


def interpolate_cav_values(frame_times, x_values, y_values):
    interp_y_values = []
    idx = 10  # Starting index in the x_values and y_values
    for time in frame_times:
        while idx < len(x_values) - 1 and x_values[idx] <= time:
            idx += 1
        start_idx = max(0, idx - 30)
        end_idx = min(len(y_values), idx + 31)
        average_y = np.mean(y_values[start_idx:end_idx])
        interp_y_values.append(average_y)

    return interp_y_values


def get_fitting_stress(fps, disk, csv_filename):
    x_values = []
    y_values = []

    with open(csv_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            time = float(row[3])  # Assuming time is in the first column
            value = float(row[4])  # Assuming value is in the second column
            x_values.append(time)
            y_values.append(value)

    # Calculate the time values for the video frames using the frame rate (fps)
    num_frames = len(disk)
    frame_times = np.linspace(0, num_frames / fps, num_frames)

    # Convert the video values to a numpy array
    video_values = np.array(disk)

    # Calculate the derivative of the video values
    video_derivative = np.gradient(video_values)

    # Calculate the derivative of the CSV values
    csv_derivative = np.gradient(y_values)

    csv_derivative = np.resize(csv_derivative, len(video_derivative))
    # Find the position where both derivatives start to change
    sync_start = max(
        0,
        np.where(
            (np.abs(video_derivative) > 0) & (np.abs(csv_derivative) > 0)
        )[0][0],
    )

    # Find the position where both derivatives stop changing
    sync_stop = min(len(video_derivative), len(csv_derivative))

    # Sync the video values and CSV values based on the found positions
    synced_video_values = video_values[sync_start:340]
    synced_frame_times = frame_times[sync_start:340]
    synced_csv_values = y_values[sync_start:340]

    # Interpolate the CSV values for the synced frame times
    interp_csv_values = interpolate_cav_values(
        synced_frame_times, x_values, y_values
    )

    interp_csv_values = [(-1) * i for i in interp_csv_values]

    # Plot the synced values
    plt.plot(interp_csv_values, synced_video_values, marker=".", linestyle="")
    plt.xlabel("stress")
    plt.ylabel("avg pixel value")
    plt.legend(["Video", "CSV"])
    plt.grid(True)
    plt.show()


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
    (disks, grad_disks) = plot_func(path_list, circle_num, debug_mode=debug)
    if args.plot_all:
        plot_all_disk_average_values(disks, plot_axis=(3, 3))
        plot_all_disk_average_values(grad_disks, plot_axis=(3, 3))

    else:
        plot_disk_average_values(disks)
        plot_disk_average_values(grad_disks)
