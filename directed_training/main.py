from __future__ import print_function, division
from model import get_model
from mnist_test import get_mnist, order_dataset
from keras.callbacks import LambdaCallback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


records = []


batch_print_callback = LambdaCallback(
    on_batch_end=lambda batch,logs: records.append([logs["loss"],logs["acc"]])
)


def main():
    global records
    x_train, y_train, x_test, y_test = get_mnist()
    training_iterations = 2
    batch_size = 128

    n_datapoints = len(x_train)//batch_size+1
    acc_history = [0 for _ in range(n_datapoints)]
    loss_history = [0 for _ in range(n_datapoints)]

    for _ in range(training_iterations):
        model = get_model()
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  callbacks=[batch_print_callback],
                  validation_data=(x_test, y_test))
        # score = model.evaluate(x_test, y_test, verbose=0)

        for i,[l,a] in enumerate(records):
            acc_history[i] += a
            loss_history[i] += l

        records = []

    acc_history = np.asarray(acc_history)/training_iterations
    loss_history = np.asarray(loss_history)/training_iterations

    fig, ax = plt.subplots()
    ax.plot(acc_history)

    ax.set(xlabel='batches (size {})'.format(batch_size), ylabel='accuracy',
           title='Accuracy vs. Training Batches')
    ax.grid()
    plt.show()



if __name__ == "__main__":
    main()

