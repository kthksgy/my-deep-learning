import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tfds_e
from data_augmentation import augment

tf.compat.v1.enable_eager_execution()

def main():
    # ~/.datasetsからデータセットを読み込み
    datasets, info = tfds_e.load('stl10', as_supervised=True)
    print(datasets)
    plt.subplots_adjust(hspace=0.75)

    ds = datasets['test'].batch(100, drop_remainder=True)
    ds = augment(ds, 16, 96, 96,
        horizontal_flip=True, vertical_flip=False,
        brightness_delta=0.2, hue_delta=0.0,
        contrast_range=[0.9, 1.2], saturation_range=[0.9, 1.2],
        width_shift=0.1, height_shift=0.1,
        rotation=15
        )
    # ds = datasets['test'].batch(100, drop_remainder=True)
    # print(ds.element_spec[0].shape)

    figure_index = 1
    file_index = 0
    for images, labels in ds:
        for image, label in zip(images, labels):
            # マップ関数を適用した後のデータが出る
            plt.subplot(2, 2, figure_index)
            plt.title(str(label.numpy()))
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            plt.imshow(np.squeeze(image.numpy().astype(np.uint8) * 6), cmap='gray')
            figure_index += 1
            if figure_index > 4:
                plt.pause(0.5)
                figure_index = 1


if __name__ == '__main__':
    main()