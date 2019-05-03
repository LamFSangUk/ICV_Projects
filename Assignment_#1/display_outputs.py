import matplotlib.pyplot as plt
from util import normal2rgb, validate_with_specific_light


def display_outputs(output_dir, num_image, img_arr, albedo_img, normal_map, light_dirs, albedo_img_un, normal_map_un, light_dirs_un):
    # save the images from photometric stereo
    plt.imshow(albedo_img, cmap='gray')
    plt.title('albedo')
    plt.savefig(output_dir + 'albedo.png')

    normal_rgb = normal2rgb(normal_map)
    plt.imshow(normal_rgb)
    plt.title('normal map')
    plt.savefig(output_dir + 'normal_map.png')

    # save the images from uncalibrated photometric stereo
    plt.imshow(albedo_img_un, cmap='gray')
    plt.title('albedo uncalibrated')
    plt.savefig(output_dir + 'albedo_uncalibrated.png')

    normal_rgb_un = normal2rgb(normal_map_un)
    plt.imshow(normal_rgb_un)
    plt.title('normal map uncalibrated')
    plt.savefig(output_dir + 'normal map_uncalibrated.png')
    plt.clf()

    plt.figure(1)
    plt.axis('off')
    plt.subplot(2, 2, 1)
    plt.imshow(albedo_img, cmap='gray')
    plt.title('albedo')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(normal_rgb)
    plt.title('normal map')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(albedo_img_un, cmap='gray')
    plt.title('albedo uncalibrated')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(normal_rgb_un)
    plt.title('normal map uncalibrated')
    plt.axis('off')
    plt.show(block=False)

    # validate photometric stereo
    plt.figure(2, figsize=(1.25 * (num_image + 2), 6))
    plt.axis('off')
    plt.suptitle('Evaluation with specific light direction')
    rows = 3
    cols = num_image + 2
    # f, axarr = plt.subplots(3, num_image+1)
    for i in range(num_image):
        re_img = validate_with_specific_light(normal_map, albedo_img, light_dirs[i])
        re_img_un = validate_with_specific_light(normal_map_un, albedo_img_un, light_dirs_un[i])

        str_light = '[' + '\n'.join(format(f, '.3f') for f in light_dirs[i]) + ']'

        # original image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_arr[i], cmap='gray')
        plt.title(str_light, fontsize=10)
        if i is 0:
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.yticks([])
            plt.ylabel('original')
        else:
            plt.axis('off')

        # calibrated photometric stereo image
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(re_img, cmap='gray')
        if i is 0:
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.yticks([])
            plt.ylabel('calibrated')
        else:
            plt.axis('off')

        # uncalibrated photometric stereo image
        plt.subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(re_img_un, cmap='gray')
        if i is 0:
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.yticks([])
            plt.ylabel('uncalibrated')
        else:
            plt.axis('off')

        # axarr[0, i+1].imshow(img_arr[i], cmap='gray')
        # axarr[1, i+1].imshow(re_img, cmap='gray')
        # axarr[2, i+1].imshow(re_img_un, cmap='gray')

    # specific light - front
    re_img = validate_with_specific_light(normal_map, albedo_img, [0, 0, 1])
    re_img_un = validate_with_specific_light(normal_map_un, albedo_img_un, [0, 0, 1])

    plt.subplot(rows, cols, 1 * cols - 2 + 1)
    plt.title('Front light\n[0, 0, 1]')
    plt.axis('off')

    plt.subplot(rows, cols, 2 * cols - 2 + 1)
    plt.imshow(re_img, cmap='gray')
    plt.axis('off')

    plt.subplot(rows, cols, 3 * cols - 2 + 1)
    plt.imshow(re_img_un, cmap='gray')
    plt.axis('off')

    # specific light - behind
    re_img = validate_with_specific_light(normal_map, albedo_img, [0, 0, -1])
    re_img_un = validate_with_specific_light(normal_map_un, albedo_img_un, [0, 0, -1])

    plt.subplot(rows, cols, 1 * cols - 1 + 1)
    plt.title('Behind light\n[0, 0, -1]')
    plt.axis('off')

    plt.subplot(rows, cols, 2 * cols - 1 + 1)
    plt.imshow(re_img, cmap='gray')
    plt.axis('off')

    plt.subplot(rows, cols, 3 * cols - 1 + 1)
    plt.imshow(re_img_un, cmap='gray')
    plt.axis('off')

    plt.savefig(output_dir + 'validation.png')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025)
    plt.show()

    return
