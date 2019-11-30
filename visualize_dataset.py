import matplotlib.pyplot as plt
import tfds_e

def main():
    # ~/.datasetsからデータセットを読み込み
    datasets, info = tfds_e.load('cifar10')
    plt.subplots_adjust(hspace=0.75)

    figure_index = 1
    file_index = 0
    for image, label in datasets['test']:
        # マップ関数を適用した後のデータが出る
        plt.subplot(8, 8, figure_index)
        plt.title(str(label.numpy()))
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(image.numpy())
        figure_index += 1
        if figure_index > 64:
            plt.pause(0.5)
            figure_index = 1


if __name__ == '__main__':
    main()