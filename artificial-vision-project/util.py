import cv2
from matplotlib import pyplot as plt
import numpy as np

#  -------------------- UTILS --------------------

def distance(
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> float:
    try:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except Exception as e:
        print(f'Error: {e}')


def gaussian(
    x: float,
    sigma: float
) -> float:
    try :
        return (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * sigma ** 2))
    except Exception as e:
        print(f'Error: {e}')


def resize_images(
    images: list[np.ndarray],
    size: tuple = (32, 32)
) -> list[np.ndarray]:
    try:
        resized_images = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]
        return resized_images
    except Exception as e:
        print(f'Error: {e}')


def convert_to_grayscale(
    images: list[np.ndarray]
) -> list[np.ndarray]:
    try:
        grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        return grayscale_images
    except Exception as e:
        print(f'Error: {e}')


def normalize_images(
    images: list[np.ndarray]
) -> list[np.ndarray]:
    try:
        normalized_images = [image.astype('float32') / 255.0 for image in images]
        return normalized_images
    except Exception as e:
        print(f'Error: {e}')

#  -------------------- COLOR SPACES --------------------


def read_images(
    images_info: list[dict]
) -> list[np.ndarray]:
    original_images = []
    original_images_gray = []
    image_names = []

    for info in images_info:
        color_image = read_image(info["path"])
        original_images.append(color_image)

        gray_image = read_image(info["path"], 'gray')
        original_images_gray.append(gray_image)

        image_names.append(info["name"])
    
    return original_images, original_images_gray, image_names


def read_image(
    image_path: str,
    mode: str = 'rgb'
) -> np.ndarray:
    try:
        grayscale_modes = ('grayscale', 'greyscale', 'gray', 'grey', 'gris')
        mode = mode.lower()
        if mode == 'rgb':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif mode == 'standard-color':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        elif mode in grayscale_modes:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif mode == 'bgr':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif mode == 'cmy':
            return 255 - cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif mode == 'yiq':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif mode == 'yuv':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)
        elif mode == 'hsv':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
        elif mode == 'hls':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HLS)
        elif mode == 'lab':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
        elif mode == 'xyz':
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2XYZ)
        else:
            raise ValueError(
                f'Modo inválido. Use "color", "standard-color", "{grayscale_modes}", "yuv", "hsv", "hls", "lab" o "xyz".')
    except Exception as e:
        print(f'Error: {e}')


def show_image(
    image: np.ndarray,
    title: str = 'Image',
    figsize: tuple = None
) -> None:
    try:
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f'Error: {e}')


def subplot_images(
    title: str = '',
    images: list = [],
    image_names: list = [],
    figsize: tuple = (15, 7)
) -> None:
    try:
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        fig.suptitle(title, fontsize=20)
        for ax, image, name in zip(axes, images, image_names):
            ax.set_title(name)
            ax.imshow(image, cmap='gray')
            ax.axis('off')
        plt.show()
    except Exception as e:
        print(f'Error: {e}')


def plot_image_and_histogram(
    image: np.ndarray,
    title: str = '',
    image_title: str = '',
    hist_title: str = '',
    figsize=(15, 4)
) -> None:
    try:
        if len(image.shape) == 3:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title, fontsize=20)
            colors = ('r', 'g', 'b')
            axs[0].set_title(image_title)
            axs[0].imshow(image, cmap='gray')
            axs[0].axis('off')

            axs[1].set_title(hist_title)
            for i, col in enumerate(colors):
                img_array_i = image[:, :, i].ravel()
                axs[1].hist(
                    img_array_i,
                    histtype='step',
                    bins=255,
                    range=(0.0, 255.0),
                    density=True,
                    color=colors[i]
                )
            plt.show()
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title, fontsize=20)
            axs[0].set_title(image_title)
            axs[0].imshow(image, cmap='gray')
            axs[0].axis('off')

            axs[1].set_title(hist_title)
            axs[1].hist(
                image.ravel(),
                bins=255,
                range=(0.0, 255.0),
                density=True
            )
            plt.show()
    except Exception as e:
        print(f'Error: {e}')


def plot_channels(
    image: np.ndarray,
    mode='rgb',
    title: str = 'RGB channels',
    channel_names: tuple = ['Channel R', 'Channel G', 'Channel B'],
    cmaps: tuple = ('Reds', 'Greens', 'Blues'),
    figsize: tuple = (30, 7),
    individual_images: list = []
) -> None:
    try:
        mode = mode.lower()
        if mode == 'rgb':
            channels = [image[:, :, i] for i in range(3)]
        elif mode == 'bgr':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = cmaps[::-1]
            channel_names = channel_names[::-1]
        elif mode == 'cmy':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('GnBu', 'RdPu', 'YlOrBr')
            channel_names = ('Channel C', 'Channel M', 'Channel Y')
        elif mode == 'yiq':
            channels = [
                0.299 * image[:, :, 0] + 0.587 *
                image[:, :, 1] + 0.114 * image[:, :, 2],
                0.596 * image[:, :, 0] - 0.274 *
                image[:, :, 1] - 0.322 * image[:, :, 2],
                0.211 * image[:, :, 0] - 0.523 *
                image[:, :, 1] + 0.312 * image[:, :, 2]
            ]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ('Channel Y', 'Channel I', 'Channel Q')
        elif mode == 'yuv':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ('Channel Y', 'Channel U', 'Channel V')
        elif mode == 'hls':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel H', 'Channel L', 'Channel S']
        elif mode == 'hsv':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel H', 'Channel S', 'Channel V']
        elif mode == 'lab':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel L', 'Channel A', 'Channel B']
        elif mode == 'xyz':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = ('gray', 'gray', 'gray')
            channel_names = ['Channel X', 'Channel Y', 'Channel Z']
        elif mode == 'custom':
            channels = [image[:, :, i] for i in range(3)]
            cmaps = cmaps
            channel_names = channel_names
        elif mode == 'individual':
            channels = individual_images
            cmaps = cmaps
            channel_names = channel_names
        else:
            raise ValueError(
                f'Invalid mode. Use "rgb", "bgr", "cmy", "yiq", "yuv", "hsl", "hsv", "lab", "xyz", "custom" or "individual".')
        # Plot the channels
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=20)

        for ax, channel, name, cmap in zip(axes, channels, channel_names, cmaps):
            ax.set_title(name)
            ax.imshow(channel, cmap=cmap, aspect='auto')
            ax.axis('off')
    except Exception as e:
        print(f'Error: {e}')

#  --------- BASIC PIXEL TO PIXEL TRANSFORMATIONS ---------


def linear_transformation(
    image: np.ndarray,
    alpha: float,
    beta: float,
    mode='default'
) -> np.ndarray:
    try:
        img = image.copy()
        if mode == 'default':
            return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        elif mode == 'grayscale':
            return cv2.add(cv2.multiply(img, alpha), beta)
        elif mode == 'color':
            size = np.shape(img)
            for i in range(size[0]):
                for j in range(size[1]):
                    for k in range(size[2]):
                        value = img[i][j][k] * alpha + beta
                        if value > 255:
                            img[i][j][k] = 255
                        elif value < 0:
                            img[i][j][k] = 0
                        else:
                            img[i][j][k] = value
            return img
        else:
            raise ValueError(
                f'Invalid mode. Use "default", "grayscale" or "color".')
    except Exception as e:
        print(f'Error: {e}')
        return None


def negative_transformation(
    image: np.ndarray
) -> np.ndarray:
    try:
        return linear_transformation(image, -1, 255)
    except Exception as e:
        print(f'Error: {e}')

# -------------------- GEOMETRIC TRANSFORMATIONS --------------------


def translation_transformation(
    image: np.ndarray,
    dx: int,
    dy: int
) -> np.ndarray:
    try:
        size = np.shape(image)
        translation_matrix = np.float32(
            [[1, 0, dx],
            [0, 1, dy]]
        )
        return cv2.warpAffine(image.copy(), translation_matrix, (size[1], size[0]))
    except Exception as e:
        print(f'Error: {e}')


def reflection_transformation(
    image: np.ndarray,
    mode: str
) -> np.ndarray:
    try:
        size = np.shape(image)
        if mode == 'x':
            reflection_matrix = np.float32(
                [[-1, 0, size[1]],
                [0, 1, 0]]
            )
        elif mode == 'y':
            reflection_matrix = np.float32(
                [[1, 0, 0],
                [0, -1, size[0]]]
            )
        elif mode == 'xy':
            reflection_matrix = np.float32(
                [[-1, 0, size[1]],
                [0, -1, size[0]]]
            )
        else:
            raise ValueError(f'Invalid mode. Use "x", "y" or "xy".')
        return cv2.warpAffine(image.copy(), reflection_matrix, (size[1], size[0]))
    except Exception as e:
        print(f'Error: {e}')


def rotation_transformation(
    image: np.ndarray,
    angle: float
) -> np.ndarray:
    try:
        size = np.shape(image)
        rotation_matrix = cv2.getRotationMatrix2D(
            (size[1] / 2, size[0] / 2), angle, 1)
        return cv2.warpAffine(image.copy(), rotation_matrix, (size[1], size[0]))
    except Exception as e:
        print(f'Error: {e}')


def inclination_transformation(
    image: np.ndarray,
    incl_x: float,
    incl_y: float
) -> np.ndarray:
    try:
        size = np.shape(image)
        inclination_matrix = np.float32(
            [[1, incl_x, 0],
            [incl_y, 1, 0]]
        )
        return cv2.warpAffine(image.copy(), inclination_matrix, (size[1], size[0]))
    except Exception as e:
        print(f'Error: {e}')


def scaling_transformation(
    image: np.ndarray,
    size: tuple = None,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    method: str = 'linear'
) -> np.ndarray:
    try:
        if method == 'linear':
            return cv2.resize(image.copy(), size, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
        elif method == 'nearest':
            return cv2.resize(image.copy(), size, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)
        elif method == 'bilinear':
            return cv2.resize(image.copy(), size, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
        elif method == 'bicubic':
            return cv2.resize(image.copy(), size, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError(
                f'Invalid method. Use "nearest", "linear", "bilinear" or "bicubic".')
    except Exception as e:
        print(f'Error: {e}')

#  -------------------- IMAGE TRANSFORMATIONS --------------------


def apply_transformation(
    image: np.ndarray,
    transformation: callable,
    args: list
) -> np.ndarray:
    try:
        if len(image.shape) == 2:
            return transformation(image.copy(), *args)
        transformed_image = np.zeros(image.shape, np.uint8)
        for i in range(3):
            transformed_image[:, :, i] = transformation(image.copy()[:, :, i], *args)
        return transformed_image
    except Exception as e:
        print(f'Error: {e}')


def gamma_correction(
    image: np.ndarray,
    a: int,
    gamma: float
) -> np.ndarray:
    try:
        result_image = cv2.multiply(
            cv2.pow(image.copy().astype(np.float32) / 255.0, gamma), a)
        return np.clip(result_image * 255.0, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')

#  -------------------- HISTOGRAMS --------------------

def equalize_histogram(
    image: np.ndarray
) -> np.ndarray:
    try:
        return cv2.equalizeHist(image.copy())
    except Exception as e:
        print(f'Error: {e}')


def histogram_expansion(
    image: np.ndarray
) -> np.ndarray:
    try:
        if len(image.shape) == 2:
            min_val = np.min(image)
            max_val = np.max(image)
            expanded_image = (image - min_val) * (255.0 / (max_val - min_val))
            expanded_image = np.uint8(expanded_image)
        else:
            channels = cv2.split(image)
            expanded_channels = []
            for ch in channels:
                min_val = np.min(ch)
                max_val = np.max(ch)
                expanded_ch = (ch - min_val) * (255.0 / (max_val - min_val))
                expanded_ch = np.uint8(expanded_ch)
                expanded_channels.append(expanded_ch)
            expanded_image = cv2.merge(expanded_channels)
        return expanded_image
    except Exception as e:
        print(f'Error: {e}')


#  -------------------- IMAGE FILTERS --------------------


def apply_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    border_type: int = cv2.BORDER_CONSTANT
) -> np.ndarray:
    try:
        return cv2.filter2D(image.copy(), -1, kernel, borderType=border_type)
    except Exception as e:
        print(f'Error: {e}')


def gaussian_kernel(
    size: int,
    sigma: float
) -> np.ndarray:
    try:
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    except Exception as e:
        print(f'Error: {e}')


def box_kernel(
    size: int
) -> [int, np.ndarray]: # type: ignore
    try:
        return (size // 2, np.ones((size, size), np.float64) / (size ** 2))
    except Exception as e:
        print(f'Error: {e}')


def blur_filter(
    image: np.ndarray,
    size: int,
) -> np.ndarray:
    try:
        return cv2.blur(image.copy(), (size, size), borderType=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f'Error: {e}')


def median_filter(
    image: np.ndarray,
    size: int
) -> np.ndarray:
    try:
        return cv2.medianBlur(image.copy(), size)
    except Exception as e:
        print(f'Error: {e}')


def maximum_filter(
    image: np.ndarray,
    size: int
) -> np.ndarray:
    try:
        kernel = np.zeros(image.shape, np.uint8)
        radius = size // 2
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                windows = image[i - radius: i + radius + 1, j - radius: j + radius + 1]
                kernel[i, j] = np.max(windows)
        return kernel.astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def minimum_filter(
    image: np.ndarray,
    size: int
) -> np.ndarray:
    try:
        kernel = np.zeros(image.shape, np.uint8)
        radius = size // 2
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                windows = image[i - radius: i + radius + 1, j - radius: j + radius + 1]
                kernel[i, j] = np.min(windows)
        return kernel.astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def bilateral_filter(
    image: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float
) -> np.ndarray:
    try:
        return cv2.bilateralFilter(image.copy(), d, sigma_color, sigma_space).astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def laplacian_filter(
    image: np.ndarray,
    ksize: int
) -> np.ndarray:
    try:
        return cv2.Laplacian(image.copy(), cv2.CV_64F, ksize=ksize)
    except Exception as e:
        print(f'Error: {e}')


def gaussian_filter(
    image: np.ndarray,
    ksize: int,
    sigma: float
) -> np.ndarray:
    try:
        return cv2.GaussianBlur(image.copy(), (ksize, ksize), sigma)
    except Exception as e:
        print(f'Error: {e}')

#  -------------------- BORDER TYPES --------------------


def border_image(
    image: np.ndarray,
    upper: int,
    lower: int,
    left: int,
    right: int,
    border_type: str ='constant'
) -> np.ndarray:
    try:
        if border_type == 'constant':
            return cv2.copyMakeBorder(image.copy(), upper, lower, left, right, cv2.BORDER_CONSTANT, value=0)
        elif border_type == 'replicate':
            return cv2.copyMakeBorder(image.copy(), upper, lower, left, right, cv2.BORDER_REPLICATE)
        elif border_type == 'reflect':
            return cv2.copyMakeBorder(image.copy(), upper, lower, left, right, cv2.BORDER_REFLECT)
        elif border_type == 'reflect101':
            return cv2.copyMakeBorder(image.copy(), upper, lower, left, right, cv2.BORDER_REFLECT_101)
        elif border_type == 'wrap':
            return cv2.copyMakeBorder(image.copy(), upper, lower, left, right, cv2.BORDER_WRAP)
        else:
            raise ValueError(
                f'Invalid border type. Use "constant", "replicate", "reflect", "reflect101" or "wrap".')
    except Exception as e:
        print(f'Error: {e}')

#  ------------------------- NOISE -------------------------


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0 
) -> np.ndarray:
    try:
        noise = np.random.normal(mean, std, image.shape)
        return np.clip(image.copy() + noise, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def add_poison_noise(
    image: np.ndarray,
) -> np.ndarray:
    try:
        return np.clip(np.random.poisson(image.copy()), 0, 255).astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def add_salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05
) -> np.ndarray:
    try:
        image_noise = image.copy()
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            black = np.array([0, 0, 0], dtype=image.dtype)
            white = np.array([255, 255, 255], dtype=image.dtype)
        prob = np.random.random(image_noise.shape[:2])
        image_noise[prob < amount / 2] = black
        image_noise[prob > 1 - amount / 2] = white
        return image_noise
    except Exception as e:
        print(f'Error: {e}')


def add_speckle_noise(
    image: np.ndarray,
    amount: float = 1.0,
    distr_width: float = 1.0
) -> np.ndarray:
    try:
        image_noisy = image.copy()
        uniform_noise = np.random.uniform(-distr_width, distr_width, (image.shape)) * amount
        return np.clip(image_noisy + image_noisy * uniform_noise, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f'Error: {e}')


def reduce_noise(
    image: np.ndarray,
    h: float = 3,
    hColor: float = 3,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21
) -> np.ndarray:
    try:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                image.copy(),
                None,
                h,
                templateWindowSize,
                searchWindowSize
            )
        else: 
            return cv2.fastNlMeansDenoisingColored(
                image.copy(),
                None,
                h,
                hColor,
                templateWindowSize,
                searchWindowSize
            )
    except Exception as e:
        print(f'Error: {e}')


def enhance_resolution(
    image: np.ndarray,
    scale: int = 2,
    model_path: str = '/EDSR_x2.pb'
) -> np.ndarray:
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", scale)
        return sr.upsample(image.copy())
    except Exception as e:
        print(f'Error: {e}')


#  -------------------- SEGMENTATION --------------------


def simple_thresholding(
    image: np.ndarray,
    threshold: int,
    method: int = cv2.THRESH_BINARY
) -> np.ndarray:
    try:
        return cv2.threshold(image.copy(), threshold, 255, method)[1]
    except Exception as e:
        print(f'Error: {e}')


def adaptative_thresholding(
    image: np.ndarray,
    method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold: int = cv2.THRESH_BINARY,
    blockSize: int = 11,
    C: float = 2
) -> np.ndarray:
    try:
        return cv2.adaptiveThreshold(image.copy(), 255, method, threshold, blockSize, C)
    except Exception as e:
        print(f'Error: {e}')


def canny_edge_detection(
    image: np.ndarray,
    threshold1: int,
    threshold2: int
) -> np.ndarray:
    try:
        return cv2.Canny(image.copy(), threshold1, threshold2)
    except Exception as e:
        print(f'Error: {e}')


def dilate_image(
    image: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 1
) -> np.ndarray:
    try:
        return cv2.dilate(image.copy(), kernel, iterations=iterations)
    except Exception as e:
        print(f'Error: {e}')
