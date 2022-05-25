import copy

import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.io import imshow, show
from scipy import signal
from scipy.signal import convolve2d


def add_zeros(image):
    """
    Функция добавляет нули к изображению со всех его сторон
    :param image: входное изображение
    :return: изображение с добавленными нулями
    """
    copy_image = copy.copy(image)
    copy_image = np.insert(copy_image, 0, 0, axis=1)  # добавляем строку сверху
    copy_image = np.insert(copy_image, 0, 0, axis=0)  # добавляем строку слева

    copy_image = np.insert(copy_image, copy_image.shape[0], 0, axis=1)  # добавляем строку снизу
    copy_image = np.insert(copy_image, copy_image.shape[0], 0, axis=0)  # добавляем строку справа
    return copy_image


def correlation_method(image, mask):
    image_size = image.shape[0]
    x = copy.copy(image)
    x = add_zeros(x)
    R = np.zeros((image_size, image_size), dtype=np.float32)

    # Нахождение энергии t(mask)
    t_energy = 0
    for k in range(0, mask.shape[0]):
        for l in range(0, mask.shape[1]):
            t_energy += (mask[k][l] ** 2)

    for n in range(1, image_size + 1):
        for m in range(1, image_size + 1):
            nominator = 0  # B = x(n + k, m + l)*t(k, l)        (k, l) from D
            denominator = 0  # x_energy = {x(n + k, m + l)}^2   (k, l) from D

            for k in range(-1, 2, 1):  # {-1, 0, 1}
                for l in range(-1, 2, 1):  # {-1, 0, 1}
                    nominator += (x[n + k][m + l] * mask[k + 1][l + 1])
                    denominator += (x[n + k][m + l] ** 2)

            R[n - 1][m - 1] = nominator / np.sqrt(denominator * t_energy)

    return R


def get_image_with_objects(background, count, obj):
    copy_background = copy.copy(background)

    # Массив непересекающихся индексов для вставки объектов
    positions = np.array([[3, 3], [3, 13], [3, 23], [3, 33], [3, 43], [3, 53], [3, 60],
                          [13, 3], [13, 13], [13, 23], [13, 33], [13, 43], [13, 53], [13, 60],
                          [23, 3], [23, 13], [23, 23], [23, 33], [23, 43], [23, 53], [23, 60],
                          [33, 3], [33, 13], [33, 23], [33, 33], [33, 43], [33, 53], [33, 60],
                          [43, 3], [43, 13], [43, 23], [43, 33], [43, 43], [43, 53], [43, 60],
                          [53, 3], [53, 13], [53, 23], [53, 33], [53, 43], [53, 53], [53, 60],
                          [60, 3], [60, 13], [60, 23], [60, 33], [60, 43], [60, 53], [60, 60]], dtype=np.int32
                         )

    i = 0
    while i != count:
        idx = positions[np.random.randint(0, len(positions) - 1)]
        if copy_background[idx[0]][idx[1]] == 128:  # Если на этом месте уже есть объект, то генерируем другой
            # i -= 1
            continue
        copy_background[idx[0]][idx[1]] = 128
        i += 1

    result = convolve2d(copy_background, obj, mode='same', boundary='symm', fillvalue=0)
    return result


def thresholding(image, threshold):
    return np.where(image >= threshold, 255, 0)


def correlator(img_with_objects, object1, object2):
    img_copy = np.copy(img_with_objects)

    fig = plt.figure(figsize=(16, 10))

    noise_background = np.random.normal(0, np.sqrt(np.var(img_copy)), (64, 64)).astype(
        np.int32)  # фон - белый шум(0,sigma(obj))

    img_copy += noise_background

    fig.add_subplot(2, 3, 1)
    plt.title('Исходное изображение фона')
    imshow(noise_background, cmap='gray')

    # фон, объекты и шум
    fig.add_subplot(2, 3, 4)
    plt.title('Фон с добавленными объектами')
    imshow(img_copy, cmap='gray')

    img_correlation1 = correlation_method(img_copy, object1)
    img_correlation2 = correlation_method(img_copy, object2)

    fig.add_subplot(2, 3, 2)
    plt.title('Корреляционное поле первого объекта')
    imshow(img_correlation1, cmap='gray')

    fig.add_subplot(2, 3, 5)
    plt.title('Корреляционное поле второго объекта')
    imshow(img_correlation2, cmap='gray')

    result1 = thresholding(img_correlation1, THRESHOLD)
    result2 = thresholding(img_correlation2, THRESHOLD)

    fig.add_subplot(2, 3, 3)
    plt.title('Пороговая обработка первого корреляционного поля')
    imshow(result1, cmap='gray')

    fig.add_subplot(2, 3, 6)
    plt.title('Пороговая обработка второго корреляционного поля')
    imshow(result2, cmap='gray')

    show()


if __name__ == '__main__':
    OBJECTS_COUNT = 8
    SHAPE = (64, 64)
    THRESHOLD = 0.93
    img = np.zeros((64, 64))

    # Objects
    T = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int32)
    REV_T = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32)

    black_background = np.zeros((SHAPE[0], SHAPE[1]), dtype=np.int32)

    object_with_t = get_image_with_objects(black_background, 3, T)
    object_with_rev_t = get_image_with_objects(black_background, 3, REV_T)
    t_and_rev_t = object_with_t + object_with_rev_t

    correlator(object_with_t, T, REV_T)
    correlator(object_with_rev_t, T, REV_T)
    correlator(t_and_rev_t, T, REV_T)
