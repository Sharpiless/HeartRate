from matplotlib import pyplot as plt


class Plot:
    def plot_data(self, x, y, color, lw, x_label='', y_label='', title=''):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.plot(x, y, color=color, lw=lw)
        plt.show()
        plt.savefig('result/' + title + '.jpg')


