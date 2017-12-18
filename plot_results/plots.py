import matplotlib.pyplot as plt
import numpy as np
import seaborn

def custom_plot(x, y, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y, **kwargs)
    ax.fill_between(x, 0.90*y, 1.03*y, facecolor=base_line.get_color(), alpha=0.2)

#architecture_1 = np.load()
x = np.asarray([1,2,33, 45, 88, 98, 99, 99, 99, 98])
x_2 = np.asarray([12, 22, 21, 41, 76, 97, 98, 99, 99, 99])
y = range(len(x))

custom_plot(y, x, color='blue', lw=3)
custom_plot(y, x_2, color='green', lw=3)

plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()