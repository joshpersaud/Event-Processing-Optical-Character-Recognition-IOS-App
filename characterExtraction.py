import cv2
import os
import re
import numpy as np
import math

# global variables
output_image_size = (28, 28)
input_image_height = 1850  # Res to resize input image. Keeps the same ratio of the original image

# all supported image formats the program looks for
ext = [".png", ".PNG", ".jpeg", "jpg", ".tif", ".pbm", ".pgm", ".ppm", ".bmp", ".dib", ".sr", ".ras",
       ".jp2"]


# Create folders to store images of characters if they don't exist
def data_folder_checks():
    if not os.path.exists("Dataset/"):
        print("Making a Dataset Folder.")
        os.makedirs("Dataset/")
    if not os.path.exists("Dataset/Testing Data/"):
        print("Adding \"Testing Data\" folder to Dataset folder")
        os.makedirs("Dataset/Testing Data")

    if not os.path.exists("Dataset/Training Data/"):
        print("Adding \"Training Data\" folder to Dataset folder")
        os.makedirs("Dataset/Training Data")

    if not os.path.exists("Preprocessed_Images/"):
        print(
            "A directory named \"Preprocessed_Images\" is needed. "
            "\nCreating one. "
            "\nImages needing to be processed should be within a folder named after the character you are extracting. "
            "\nThose folders containing images should be placed in the Preprocessed_Images folder.")
        os.makedirs("Preprocessed_Images/")
        print("Quitting Program. Rerun once folders are populated")
        quit()


# Check if directories contain images
def check_for_data():
    global ext
    checker = False

    if os.path.exists("Preprocessed_Images/"):
        if any(os.scandir("Preprocessed_Images/")):
            for one in range(32, 255):
                if os.path.exists(f'Preprocessed_Images/{chr(one)}/'):
                    print("Found folder: Preprocessed_Images/", chr(one))
                    if any(File.endswith(tuple(ext)) for File in os.listdir(f'Preprocessed_Images/{chr(one)}/')):
                        print("Image(s) were found in this folder.\n")
                        checker = True
                    else:
                        print("No images in folder.\n")
            if checker is False:
                print("No Images were found. Quitting program.")
                quit()


# Will return the path of every image found in all folders as a dictionary
def get_all_image_paths():
    valid_dir = dict()
    global ext
    j = 0

    for dir_path, dir_names, files in os.walk("Preprocessed_Images/"):
        for sub_files in files:
            for file_ext in ext:
                if file_ext in sub_files:
                    j = j + 1
                    dir_names = re.sub('[\[\]]', '', str(dir_names))

                    valid_dir[j] = f"{dir_path}{dir_names}/{sub_files}"
    return valid_dir


# Resize the input image set by the global variable
def input_image_resize(image, height):
    (h, w) = image.shape[:2]
    r = height / float(h)
    output_size = (int(w * r), height)

    return cv2.resize(image, output_size)


# Resize the output image set by the global variable
def output_image_resize(image):
    global output_image_size
    return cv2.resize(image, output_image_size)


# Check if the entire character is in the out image. Returns a bool
# Ran by extract_char() automatically. No need to run this yourself.
def output_image_verification(cut_out_image):
    cut_out_height, cut_out_width = cut_out_image.shape[:2]
    print("image size", cut_out_height, cut_out_width)
    i = 0
    checker = False
    res_to_check = dict()

    # Get all pixels locations that need to be checked.
    for pixel in range(cut_out_height):
        res_to_check[i] = (0, pixel)
        i = i+1
        res_to_check[i] = (cut_out_width-1, pixel)
        i = i+1
    for pixel in range(cut_out_width):
        res_to_check[i] = (pixel, 0)
        i = i+1
        res_to_check[i] = (pixel, cut_out_height-1)
        i = i+1
    # Check the pixels in the image
    for pixels in res_to_check:
        pixel_b, pixel_g, pixel_r = cut_out_image[(res_to_check[pixels])[0], (res_to_check[pixels])[1]]
        total_intensity = (int(pixel_r) + int(pixel_g) + int(pixel_b))
        if not total_intensity == 765:
            return False
        else:
            checker = True

    return checker


# Opens all supported images, inverts them, applies blurring then contour detection
# Finds the corresponding name of the char being extracted based on it's folder name
# Saves the detected characters into a folder called "Dataset"
def extract_chars(image_paths):
    for path in image_paths:

        # Gets the name of the character based on the folders name it is in
        char_name = re.sub('Preprocessed_Images/', '', image_paths[path])
        char_name, sep, file_name = char_name.partition('/')
        file_name = (file_name.rsplit(".", 1)[0])

        # Certain characters are special and a folder cannot be named after it
        # Names the folder the word of the character to get around this
        if char_name is ':':
            char_name = 'colon'
        if char_name is '.':
            char_name = 'dot'
        if char_name is '|':
            char_name = 'or'

        print("\nWorking on file:", file_name)
        print("Char found in filename is:", char_name, '\n')

        # Reads and resizes the image and inverts it
        image = cv2.imread(str(image_paths[path]))
        image = input_image_resize(image, 1850)
        inverted_image = cv2.bitwise_not(image)

        # Converts RGB image to binary and applies blurring
        image_binary = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_binary, 127, 255, 0)
        thresh = cv2.GaussianBlur(thresh, (15,15),8)
        kernel = np.ones((9, 9), np.float32) / 225
        thresh = cv2.filter2D(thresh, -1, kernel)

        # Finds all contours(characters) in the image
        print('Getting contours of the image.')
        try:
            img_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_contours_array = img_contours[0]
        except:
            print('No contours found, checking next image if one exist')
            continue

        # Makes sure a directory named after the character type exist and makes one if not
        try:
            print(f'Checking if directory \"Dataset/Training Data/{char_name}\" exist')
            os.makedirs(f"Dataset/Training Data/{char_name}")
        except:
            pass

        print("All checks passed! \nStarting Extraction :)")
        # Loop counter
        i = 0

        # Repeats for every contour(character) detected
        for contours in img_contours_array:
            i = i+1
            contour_moments = cv2.moments(contours)
            cX = int(contour_moments["m10"] / contour_moments["m00"])
            cY = int(contour_moments["m01"] / contour_moments["m00"])

            # gets area of the contour to make an approximation on the size of the box needed to fill the char.
            area = cv2.contourArea(contours)
            print("\nArea of:", i, ": ", area)

            area_adjustment = (area + 200)
            area_adjustment = math.sqrt(area_adjustment)

            # draw the contour and center point of the contour on the image
            cv2.drawContours(inverted_image, [contours], -1, (0, 255, 0), 1)
            cv2.circle(inverted_image, (cX, cY), 7, (255, 0, 0), 3)
            cv2.putText(inverted_image, f'{i}', (cX - 25, cY - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculates the size needed for the drawing of the rectangle around the contour
            square_size_real = int((area_adjustment/2))

            # Draws rectangle around the contour
            cv2.rectangle(inverted_image,
                          (cX - square_size_real, cY - square_size_real),
                          (cX + square_size_real, cY + square_size_real), (0, 0, 255), 3)

            # show the image inverted with contours and drawings
            print('Char:', i, 'of', len(img_contours_array))
            cv2.imshow("Image", inverted_image)
            cv2.waitKey(1)

            # Gets the individual character that was detected as a image
            cut_out = image[cY - square_size_real:cY + square_size_real,
                            cX - square_size_real:cX + square_size_real]

            # Adjust cutout of character if it is not fully in the frame
            # Uses our adjustment_bool() to check the image
            z = 0
            k = 0
            area_adjustment_2 = area_adjustment
            print('checking image: In bounds:', output_image_verification(cut_out))
            adjustment_bool = False
            while output_image_verification(cut_out) is False:
                adjustment_bool = True
                print('Adjusting image')
                z = z+1
                cut_out_height, cut_out_width = cut_out.shape[:2]
                k = k + 400
                if z < (cut_out_height * 4):
                    area_adjustment = math.sqrt(area_adjustment_2 + k)
                    square_size_real = int((area_adjustment / 2))
                    cut_out = image[cY - square_size_real:cY + square_size_real,
                              cX - square_size_real:cX + square_size_real]
                else:
                    print('could not adjust image.')
                    break
            if adjustment_bool:
                cv2.rectangle(inverted_image,
                              (cX - square_size_real, cY - square_size_real),
                              (cX + square_size_real, cY + square_size_real), (0, 255, 0), 2)
            # Resizes the character cutout based on the global variable
            try:
                cut_out_resize = output_image_resize(cut_out)
                cv2.imshow('Current Char', cut_out)
                cv2.waitKey(1)
                pass
            except:
                print("Out of bounds, skipping")
                continue

            # Saves the character to a folder based on its type
            try:
                print('Attempting to saving char: ', i)
                cv2.imwrite(f'Dataset/Training Data/{char_name}/{char_name}_{file_name}_{i}.png', cut_out_resize)
                print('Saved')
            except:
                print('An error occurred while saving char: ', i)
                continue


# Runs the program
data_folder_checks()
check_for_data()
image_paths = get_all_image_paths()
print("Total Images:", len(image_paths))

extract_chars(image_paths)

print("\nAll done :)")
cv2.waitKey(0)
cv2.destroyAllWindows()