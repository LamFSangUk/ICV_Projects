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

    #light_dirs = -light_dirs
    #light_dirs = light_dirs[:, [1, 2, 0]]

    return ambient_img, img_arr, light_dirs


def crop(img, x, y, width, height):
    return img[:, y:y+height, x:x+width]


def normal2rgb(normal_map):
    width = normal_map.shape[1]
    height = normal_map.shape[0]

    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            rgb = normal_map[i][j] / 2
            rgb += 0.5
            rgb *= 255

            img[i][j] = np.uint8(rgb)

    return img


def validate_with_specific_light(normal_map, albedo_img, light_dir):
    img = np.matmul(normal_map, light_dir)
    img = np.multiply(albedo_img, img)

    img *= 255
    img[img < 0] = 0

    return img


if __name__ == '__main__':
    num_images = 11
    ambient_img, img_arr, light_dirs = load_face_images('./data/', 'yaleB01', num_images)
