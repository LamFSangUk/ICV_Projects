import numpy as np
import os


def sph2cart(r, azimuth, elevation):
    return [
        r * np.sin(elevation) * np.cos(azimuth),
        r * np.sin(elevation) * np.sin(azimuth),
        r * np.cos(elevation)
    ]


def get_pgm_raw(filename):
    # Open file
    with open(filename, 'rb') as file:
        # confirm that the file is PGM
        code = file.readline()

        _ = file.readline()

        # get width, height
        width, height = file.readline().split()
        width = int(width)
        height = int(height)

        # get max gray value
        max_gray = file.readline()

        # read actual data
        img = np.fromfile(file, dtype=np.uint8, count=width * height)
        img = np.reshape(img, (height, -1))

    return img, max_gray, width, height


def load_face_images(pathname, subject_name, num_images):

    # get ambient image
    filename = pathname + subject_name + '_P00_Ambient.pgm'
    ambient_img, _, width, height = get_pgm_raw(filename)

    # get list of all other image files
    file_list = os.listdir(pathname)

    # arrays to store the angles of light source
    angle_list = np.zeros((num_images, 2))

    # create array of illuminated images
    img_arr = np.zeros((num_images, height, width))

    for i in range(num_images):
        assert(file_list[i].startswith(subject_name) is True)
        filename = file_list[i][len(subject_name):]
        idx = filename.find('A') + 1
        angle_list[i][0] = int(filename[idx:idx+4])
        idx = filename.find('E') + 1
        angle_list[i][1] = int(filename[idx:idx+3])

        filename = pathname + subject_name + filename
        img_arr[i], _, _, _ = get_pgm_raw(filename)

    # converse coordinate
    light_dirs = np.zeros((num_images, 3))
    for i in range(num_images):
        light_dirs[i] = sph2cart(1, angle_list[i][0]*np.pi/180, angle_list[i][1]*np.pi/180)  # radius, azimuth, elevation

    return ambient_img, img_arr, light_dirs


if __name__ == '__main__':
    num_images = 11
    ambient_img, img_arr, light_dirs = load_face_images('./data/', 'yaleB01', num_images)
