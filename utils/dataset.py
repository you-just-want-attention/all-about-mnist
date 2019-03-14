import numpy as np
import pandas as pd
import cv2
import os

"""
# All about MNIST Style DataSet

> We include the following dataset list
    - mnist : handwritten digits dataset

    - fashionmnist : dataset of Zalando's article images

    - handwritten : handwritten a ~ z Alphabet dataset


Current Implemented Generator list

    1. SerializationDataset

    2. CalculationDataset

    3. ClassificationDataset

    4. LocalizationDataset

    5. DetectionDataset

"""
DOWNLOAD_URL_FORMAT = "https://s3.ap-northeast-2.amazonaws.com/pai-datasets/all-about-mnist/{}/{}.csv"
DATASET_DIR = "../datasets"


class SerializationDataset:
    """
    generate data for Serialization

    이 class는 단순히 숫자를 나열하는 것

    :param dataset : Select one, (mnist, fashionmnist, handwritten)
    :param data_type : Select one, (train, test, validation)
    :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value), the length of number will be determined within the range
    :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
    :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
    """

    def __init__(self, dataset="mnist", data_type="train",
                 digit=5, bg_noise=(0, 0.2), pad_range=(3, 30)):
        """
        generate data for Serialization

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        :param digit : the length of number
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value), the length of number will be determined within the range
        :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
        :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit + 1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1] - 1)
        self.index_list = np.arange(len(self.labels))

        self.bg_noise = bg_noise
        self.pad_range = pad_range

        self.max_length = int((20 + pad_range[1]) * self.digit_range[1] * 2)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1] - 1) * index
            digits = self.index_list[start_index:start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            series_image, series_len = self._serialize_random(digit_images)

            return series_image, digit_labels, series_len

        else:
            batch_images, batch_labels, batch_length = [], [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index:start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values
                series_image, series_len = self._serialize_random(digit_images)
                batch_images.append(series_image)
                batch_labels.append(digit_labels)
                batch_length.append(series_len)

            return np.stack(batch_images), \
                np.stack(batch_labels), \
                np.stack(batch_length)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _serialize_random(self, images):
        """
        복수의 이미지를 직렬로 붙임

        :param images:
        :return:
        """
        pad_height = images.shape[1]
        pad_width = np.random.randint(*self.pad_range)

        serialized_image = np.zeros([pad_height, pad_width])
        for image in images:
            serialized_image = self._place_random(image, serialized_image)

        full_image = np.random.normal(*self.bg_noise,
                                      size=(pad_height, self.max_length))

        if serialized_image.shape[1] < self.max_length:
            series_length = serialized_image.shape[1]
            full_image[:, :serialized_image.shape[1]] += serialized_image
        else:
            series_length = full_image.shape[1]
            full_image += serialized_image[:, :full_image.shape[1]]

        full_image = np.clip(full_image, 0., 1.)
        return full_image, series_length

    def _place_random(self, image, serialized_image):
        """
        가운데 정렬된 이미지를 떼어서 재정렬함

        :param image:
        :param serialized_image:
        :return:
        """
        x_min, x_max, _, _ = crop_fit_position(image)
        cropped = image[:, x_min:x_max]

        pad_height = cropped.shape[0]
        pad_width = np.random.randint(*self.pad_range)
        pad = np.zeros([pad_height, pad_width])

        serialized_image = np.concatenate(
            [serialized_image, cropped, pad], axis=1)
        return serialized_image


class CalculationDataset:
    """
    generate data for Calculation

    이 class는 아래과 같은 수식을 automatically 만들어줌
    (1 + 2) * 3 + 5 * (2 + 3)

    :param data_type : Select one, (train, test, validation)
    :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
    :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
    :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
    """

    def __init__(self, data_type="train", digit=(5, 15),
                 bg_noise=(0, 0.2), pad_range=(3, 30)):
        """
        generate data for Calculation

        :param data_type: Select one, (train, test, validation)
        :param digit_range : the length of number (몇개의 숫자를 serialize할 것인지 결정)
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value), the length of number will be determined within the range
        :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
        :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
        """
        self.images, self.labels = load_dataset("mnist", data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit + 1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1] - 1)
        self.index_list = np.arange(len(self.labels))

        self.bg_noise = bg_noise
        self.pad_range = pad_range

        self.max_length = int((20 + pad_range[1]) * self.digit_range[1] * 2)
        self._setup_ops_image()  # create mnist-style image of brackets and operations

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1] - 1) * index
            digits = self.index_list[start_index:start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            eq_image, eq_result, equation = \
                self._create_equation_random(digit_images, digit_labels)
            series_image, series_len = self._serialize_random(eq_image)

            return series_image, eq_result, series_len, equation
        else:
            batch_images, batch_eq_results, batch_series_lens, batch_equations = \
                [], [], [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index:start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values
                eq_image, eq_result, equation = \
                    self._create_equation_random(digit_images, digit_labels)
                series_image, series_len = self._serialize_random(eq_image)
                batch_images.append(series_image)
                batch_eq_results.append(eq_result)
                batch_series_lens.append(series_len)
                batch_equations.append(equation)

            return np.stack(batch_images), \
                np.stack(batch_eq_results), \
                np.stack(batch_series_lens), \
                np.stack(batch_equations)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _create_equation_random(self, images, labels):
        """
        랜덤 수식을 만듦

        :param images:
        :param labels:
        :return:
        """
        N = len(labels)
        numbers = labels  # 숫자 N개 도출

        # 괄호 후보군을 포함한 수식 전체 리스트 만들기
        cal_series = np.array([""] * (4 * N), dtype="<U{}".format(N))

        # 숫자와 연산자 채워넣기
        cal_series[1::4] = numbers
        ops = np.random.choice(["+", "-", "*", "/"], len(cal_series[3::4]))
        cal_series[3::4] = ops

        # 괄호의 숫자 N개 도출
        num_bracket = np.random.randint(1, np.ceil(N / 2) + 1)

        for _ in range(num_bracket):
            # 왼 괄호 위치 결정
            lb_candidate = np.random.randint(0, N)
            # 오른 괄호 위치 결정 -> 왼 괄호보다는 오른쪽에 있어야함
            rb_candidate = np.random.randint(lb_candidate, N)
            # 왼괄호/오른괄호 넣기
            cal_series[lb_candidate * 4] = cal_series[lb_candidate * 4] + "("
            cal_series[rb_candidate * 4 +
                       2] = cal_series[rb_candidate * 4 + 2] + ")"

        equation = "".join(list(cal_series[:-1]))
        eq_image = self._draw_equation(images, equation)
        try:
            eq_result = eval(equation)
        except ZeroDivisionError as e:
            # TODO : ZeroDivisionError가 나오지 않는 equation generator를 만들어야 함
            # 직접 수식에서 ZeroDivisionCase를 찾는 방법이 당장 떠오르지 않음
            eq_image, eq_result, equation = self._create_equation_random(
                images, labels)

        return eq_image, eq_result, equation

    def _serialize_random(self, images):
        """
        복수의 이미지를 직렬로 붙임

        :param images:
        :return:
        """
        pad_height = images.shape[1]
        pad_width = np.random.randint(*self.pad_range)

        serialized_image = np.zeros([pad_height, pad_width])
        for image in images:
            serialized_image = self._place_random(image, serialized_image)

        full_image = np.random.normal(*self.bg_noise,
                                      size=(pad_height, self.max_length))

        if serialized_image.shape[1] < self.max_length:
            series_length = serialized_image.shape[1]
            full_image[:, :serialized_image.shape[1]] += serialized_image
        else:
            series_length = full_image.shape[1]
            full_image += serialized_image[:, :full_image.shape[1]]

        full_image = np.clip(full_image, 0., 1.)
        return full_image, series_length

    def _place_random(self, image, serialized_image):
        """
        가운데 정렬된 이미지를 떼어서 재정렬함

        :param image:
        :param serialized_image:
        :return:
        """
        x_min, x_max, _, _ = crop_fit_position(image)
        cropped = image[:, x_min:x_max]

        pad_height = cropped.shape[0]
        pad_width = np.random.randint(*self.pad_range)
        pad = np.zeros([pad_height, pad_width])

        serialized_image = np.concatenate(
            [serialized_image, cropped, pad], axis=1)
        return serialized_image

    def _setup_ops_image(self):
        # 왼괄호가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, "(", (10, 18), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, 255)
        left_bracket = cv2.GaussianBlur(image, (3, 3), 1)
        self.left_bracket = left_bracket / 255

        # 오른괄호가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, ")", (10, 18), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, 255)
        right_bracket = cv2.GaussianBlur(image, (3, 3), 1)
        self.right_bracket = right_bracket / 255

        # 더하기가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, "+", (5, 18), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, 255)
        plus = cv2.GaussianBlur(image, (3, 3), 1)
        self.plus = plus / 255

        # 빼기가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, "-", (5, 18), cv2.FONT_HERSHEY_DUPLEX,
                            0.6, 255)
        minus = cv2.GaussianBlur(image, (3, 3), 1)
        self.minus = minus / 255

        # 곱하기가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, "X", (7, 20), cv2.FONT_HERSHEY_TRIPLEX,
                            0.6, 255)
        multiply = cv2.GaussianBlur(image, (3, 3), 1)
        self.multiply = multiply / 255

        # 나누기가 잘 만들어지는지 확인
        blank = np.zeros((28, 28), np.uint8)
        image = cv2.putText(blank, "%", (7, 20), cv2.FONT_HERSHEY_TRIPLEX,
                            0.6, 255)
        divide = cv2.GaussianBlur(image, (3, 3), 1)
        self.divide = divide / 255

    def _draw_equation(self, images, equation):
        n_idx = 0
        equation_images = np.zeros((len(equation), *images.shape[1:]))
        for idx, element in enumerate(equation):
            if element.isnumeric():
                equation_images[idx] = images[n_idx]
                n_idx += 1
            elif element == "(":
                equation_images[idx] = self.left_bracket
            elif element == ")":
                equation_images[idx] = self.right_bracket
            elif element == "+":
                equation_images[idx] = self.plus
            elif element == "-":
                equation_images[idx] = self.minus
            elif element == "*":
                equation_images[idx] = self.multiply
            elif element == "/":
                equation_images[idx] = self.divide

        return equation_images


class ClassificationDataset:
    def __init__(self, dataset="mnist", data_type="train"):
        """
        generate data for classification

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        self.num_data = len(self.labels)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        batch_images = self.images[index]
        batch_labels = self.labels[index]

        if batch_images.ndim == 3:
            """
            # index > 1 -> need to stack, series to numpy.ndarray
            """
            return np.stack(batch_images), batch_labels.values
        else:
            """
            # index == 1 -> no need to stack
            """
            return batch_images, batch_labels

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))


class LocalizationDataset:
    def __init__(self, dataset="mnist", data_type="train",
                 rescale_ratio=(.8, 3.),
                 bg_size=(112, 112), bg_noise=(0, 0.2)):
        """
        generate data for localization

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        self.num_data = len(self.labels)

        self.rescale_ratio = rescale_ratio
        self.bg_size = bg_size
        self.bg_noise = bg_noise

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        batch_images = self.images[index]
        batch_labels = self.labels[index]

        if batch_images.ndim == 3:
            """
             # of index > 1 -> need to stack, series to numpy.ndarray
            """
            image_with_bgs = []
            positions = []
            labels = batch_labels.values
            for image in batch_images:
                image = self._rescale_random(image)
                image_with_bg, position = self._place_random(image)
                image_with_bgs.append(image_with_bg)
                positions.append(position)

            images = np.stack(image_with_bgs)
            positions = np.stack(positions)
            return images, positions, labels
        else:
            """
             # of index == 1 -> no need to stack
            """
            image = self._rescale_random(batch_images)
            image_with_bg, position = self._place_random(image)
            label = batch_labels
            return image_with_bg, position, label

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _rescale_random(self, image):
        value = np.random.uniform(*self.rescale_ratio)
        image = cv2.resize(image, None, fx=value, fy=value)
        return image

    def _place_random(self, image):
        background = np.random.normal(*self.bg_noise, size=self.bg_size)
        height, width = self.bg_size
        height_fg, width_fg = image.shape

        y = np.random.randint(0, height - height_fg - 1)
        x = np.random.randint(0, width - width_fg - 1)

        x_min, x_max, y_min, y_max = crop_fit_position(image)

        position = np.array([(x_min + x) / width, (x_max + x) / width,
                             (y_min + y) / height, (y_max + y) / height])

        background[y:y + height_fg, x:x + width_fg] += image

        background = np.clip(background, 0., 1.)
        return background, position


class DetectionDataset:
    def __init__(self, dataset="mnist", data_type="train",
                 digit=(3, 8), rescale_ratio=(.8, 3.),
                 bg_size=(224, 224), bg_noise=(0, 0.2)):
        """
        generate data for Detection

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value),
          the length of number will be determined within the range

        :param bg_size : the shape of background image
        :param bg_noise : the background noise of image,
               bg_noise = (gaussian mean, gaussian stddev)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit + 1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1] - 1)
        self.index_list = np.arange(len(self.labels))

        self.rescale_ratio = rescale_ratio
        self.bg_size = bg_size
        self.bg_noise = bg_noise

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1] - 1) * index
            digits = self.index_list[start_index:start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            image, digit_positions = self._scatter_random(digit_images)

            return image, digit_labels, digit_positions
        else:
            batch_images, batch_labels, batch_pos = [], [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index:start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values

                image, digit_positions = self._scatter_random(digit_images)

                batch_images.append(image)
                batch_labels.append(digit_labels)
                batch_pos.append(digit_positions)

            return np.stack(batch_images), \
                np.stack(batch_labels), \
                np.stack(batch_pos)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _scatter_random(self, images):
        background = np.random.normal(*self.bg_noise,
                                      size=self.bg_size)
        positions = []

        for image in images:
            image = self._rescale_random(image)
            background, position = self._place_random(image,
                                                      background,
                                                      positions)
            positions.append(position)

        return background, np.array(positions)

    def _rescale_random(self, image):
        value = np.random.uniform(*self.rescale_ratio)
        image = cv2.resize(image, None, fx=value, fy=value)
        return image

    def _place_random(self, image, background, prev_pos):
        height, width = self.bg_size
        height_fg, width_fg = image.shape

        x_min, x_max, y_min, y_max = crop_fit_position(image)

        while True:
            y = np.random.randint(0, height - height_fg - 1)
            x = np.random.randint(0, width - width_fg - 1)

            position = np.array([(x_min + x) / width, (x_max + x) / width,
                                 (y_min + y) / height, (y_max + y) / height])

            if not self._check_overlap(position, prev_pos):
                # 이전의 object랑 겹치지 않으면 넘어감
                break

        background[y:y + height_fg, x:x + width_fg] += image

        background = np.clip(background, 0., 1.)
        return background, position

    def _check_overlap(self, curr_pos, prev_pos):
        if len(prev_pos) == 0:
            return False
        else:
            prev_pos = np.array(prev_pos)

        # 각 면적 구하기
        curr_area = (curr_pos[1] - curr_pos[0]) * (curr_pos[3] - curr_pos[2])
        prev_area = (prev_pos.T[1] - prev_pos.T[0]) * (prev_pos.T[3] - prev_pos.T[2])

        # Intersection 면적 구하기
        _, it_min_xs, _, it_min_ys = np.minimum(curr_pos, prev_pos).T
        it_max_xs, _, it_max_ys, _ = np.maximum(curr_pos, prev_pos).T

        it_width = ((it_min_xs - it_max_xs) > 0) * (it_min_xs - it_max_xs)
        it_height = ((it_min_ys - it_max_ys) > 0) * (it_min_ys - it_max_ys)

        intersection = (it_width * it_height)
        # 전체 면적 구하기
        union = (curr_area + prev_area) - intersection
        # IOU가 5%이상이 되는 겹침 현상 발생하면, 겹쳤다고 판정
        return np.max(intersection / union) >= 0.05


def crop_fit_position(image):
    """
    get the coordinates to fit object in image

    :param image:
    :return:
    """
    positions = np.argwhere(
        image >= 0.1)  # set the threshold to 0.1 for reducing the noise

    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    return np.array([x_min, x_max, y_min, y_max])


def load_dataset(dataset, data_type):
    """
    Load the MNIST-Style dataset
    if you don't have dataset, download the file automatically

    :param dataset: Select one, (mnist, fashionmnist, handwritten)
    :param data_type: Select one, (train, test, validation)
    :return:
    """
    if dataset not in ["mnist", "fashionmnist", "handwritten"]:
        raise ValueError(
            "allowed dataset: mnist, fashionmnist, handwritten")
    if data_type not in ["train", "test", "validation"]:
        raise ValueError(
            "allowed data_type: train, test, validation")

    file_path = os.path.join(
        DATASET_DIR, "{}/{}.csv".format(dataset, data_type))

    if not os.path.exists(file_path):
        import wget
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        url = DOWNLOAD_URL_FORMAT.format(dataset, data_type)
        wget.download(url, out=file_path)

    df = pd.read_csv(file_path)

    images = df.values[:, 1:].reshape(-1, 28, 28)
    images = images / 255  # normalization, 0~1
    labels = df.label  # label information
    return images, labels


if __name__ == '__main__':
    pass
