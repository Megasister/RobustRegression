import numpy as np
from random import sample
import math
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from mpl_toolkits import mplot3d


def external_func1(linear_comb):
    return (2 * np.sin(linear_comb) + 0.5*np.power(linear_comb, 2) - linear_comb)


def external_func2(linear_comb):
    return (2 * np.cos(linear_comb) + 0.5*np.power(linear_comb, 1.5) - linear_comb)


def external_func3(linear_comb):
    return (linear_comb)


def loading_Data():
    np.random.seed(5001)
    #tf.set_random_seed(1001)

    # Genrating random data following curve equation
    # 1000 training points and 50 test points
    p, dim = 5, 5
    train_sample_size = 1000
    test_sample_size = 100
    x1 = np.linspace(0, 5, train_sample_size)
    x_test = np.linspace(0, 5, test_sample_size)
    X = np.linspace(0, 5, train_sample_size)
    X_test = np.linspace(0, 5, test_sample_size)

    while p!= 1:
        train = np.linspace(0, 5, train_sample_size)
        test = np.linspace(0, 5, test_sample_size)
        X = np.vstack((X, train))
        X_test = np.vstack((X_test, test))
        p -= 1

    dumb_row_train = np.ones((1, train_sample_size))
    dumb_row_test = np.ones((1, test_sample_size))
    X = np.vstack((X, dumb_row_train))
    X_test = np.vstack((X_test, dumb_row_test))

    print(X)
    print(X.shape)
    
    '''x1 = np.linspace(0, 5, 1000)
    x2 = np.linspace(0, 5, 1000)
    x3 = np.linspace(0, 5, 1000)
    X = np.vstack((x1, x2))

    # y = np.linspace(0, 50, 100)
    x_test = np.linspace(0, 5, 50)
    x_2 = np.linspace(0, 5, 50)
    x_3 = np.linspace(0, 5, 50)
    X_test = np.vstack((x_test, x_2))'''

    theta = np.random.uniform(1.0, 2.0, size=(1, dim+1))[0]
    print('theta is:', theta)

    multivar_train = np.dot(theta, X)
    multivar_test = np.dot(theta, X_test)

    y_o = external_func1(multivar_train)
    y_test = external_func1(multivar_test)

    # Adding uniform noise to y value of the original data conformed strictly to equation
    # np.random.seed(1)

    # normal_noise_amptitude = 10
    # y_o += np.random.uniform(-normal_noise_amptitude, normal_noise_amptitude, x1.shape[0])
    # y_test += np.random.uniform(-normal_noise_amptitude, normal_noise_amptitude, x_test.shape[0])

    # Adding normal distributed noise to only the t raining data
    np.random.seed(1)

    alphas = [0.25]
    mu1, scale1 = 0, 10
    mu2, scale2 = 100, 0.01
    noise0 = []
    noise1 = []
    noise2 = []
    noise3 = []
    y = []
    for i, alpha in enumerate(alphas):
        noise0.append(np.random.normal(mu1, scale1, size=len(x1)))
        noise1.append(np.random.normal(mu1, scale1, size=int((1 - alpha) * len(x1))))
        noise2.append(np.random.normal(mu2, scale2, size=int(alpha * len(x1))))
        noise3.append(np.random.normal(mu2, scale2, size=len(x1)))

        y.append(y_o + np.random.permutation(np.concatenate((noise1[i], noise2[i]))))

    y_sym_norm = y_o + noise0[0]
    y_asym_norm = y_o + noise3[0]

    plt.figure(1)

    # plot raw data
    plt.scatter(x1, y_o, label='no noise')
    plt.scatter(x1, y_sym_norm, label='symmetric noise')
    plt.scatter(x1, y_asym_norm, label='asymmetric noise')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Training Data")
    plt.show()

    return X, y, x1, X_test, y_test, x_test, alphas


X, Y, x1, X_test, Y_test, x_test, alphas = loading_Data()
train_sample_size = len(x1)
test_sample_size = len(x_test)


def create_placeholders():
    X_ph = tf.placeholder("float32")
    Y_ph = tf.placeholder("float32")

    return X_ph, Y_ph


def init_parameters(sample_size):
    parameters = {}

    tf.set_random_seed(1)

    xavier = tf.initializers.glorot_uniform()
    zero = tf.initializers.zeros()

    W1 = tf.Variable(xavier(shape=(500, sample_size)), dtype=tf.float32)
    b1 = tf.Variable(zero(shape=(500, 1)), dtype=tf.float32)
    W2 = tf.Variable(xavier(shape=(1, 500)), dtype=tf.float32)
    b2 = tf.Variable(zero(shape=(1, 1)), dtype=tf.float32)

    # W3 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # b3 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())

    parameters['W1'] = W1
    parameters['b1'] = b1
    parameters['W2'] = W2
    parameters['b2'] = b2
    # parameters['W3'] = W3
    # parameters['b3'] = b3

    return parameters


def forward_propation(X_ph, X_train, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    with tf.Session() as sess:
        X_ph = sess.run(X_ph, feed_dict={X_ph: X_train})
        shape = X_ph.shape[0]

    if shape > 1:
        Z1 = tf.add(tf.matmul(W1, X_ph), b1)
    else:
        Z1 = tf.add(tf.multiply(W1, X_ph), b1)

    A1 = tf.nn.leaky_relu(Z1)
    print('A1shape', A1.shape)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    # A2 = tf.nn.leaky_relu(Z2)
    # Z3 = A2*W3 + b3

    return Z2


def MSE_cost(Z2, Y):
    MSE_cost = tf.reduce_mean(tf.pow(Z2-Y, 2))
    return MSE_cost


def new_cost(Z2, Y, alp):
    new_cost = tf.reduce_mean(1 - tf.exp(-tf.pow(Z2-Y, 2) * alp/2)/alp)
    return new_cost


def model(X_train, Y_train, alp, index, learning_rate, training_epochs):

    ops.reset_default_graph()

    X_ph, Y_ph = create_placeholders()
    parameters = init_parameters(X_train.shape[0])
    Z2 = forward_propation(X_ph, X_train, parameters)

    cost = new_cost(Z2, Y_ph, alp)
    # mse_cost = MSE_cost(Z2, Y_ph)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Global Variables Initializer
    init = tf.global_variables_initializer()

    costs = []

    # Starting the Tensorflow Session
    with tf.Session() as sess:
        # Initializing the Variables
        sess.run(init)

        # Iterating through all the epochs
        for epoch in range(training_epochs):

            # Feeding each data point into the optimizer using Feed Dictionary
            # for (_x, _y) in zip(x, y):

            sess.run(optimizer, feed_dict={X_ph: X_train, Y_ph: Y_train})

            # Displaying the result after every 50 epochs
            if (epoch + 1) % 1000 == 0:
                # Calculating the cost a every epoch
                c = sess.run(cost, feed_dict={X_ph: X_train, Y_ph: Y_train})
                print("Epoch", (epoch + 1), ": cost =", c)
                costs.append(c)

            # Storing necessary values to be used outside the Session
        training_cost = sess.run(cost, feed_dict={X_ph: X_train, Y_ph: Y_train})
        # t_mse_cost = sess.run(mse_cost, feed_dict={X_ph: X_train, Y_ph: Y_train})

        print("Training cost =", training_cost, '\n')
        # print("Training mse cost =", t_mse_cost, '\n')

        new_para = sess.run(parameters, feed_dict={X_ph: X_train, Y_ph: Y_train})

        new_Z = sess.run(Z2, feed_dict={X_ph: X_train, Y_ph: Y_train})
        print(new_Z.shape)

        '''plt.figure(index)
        plt.plot(np.squeeze(costs))
        plt.xlabel('#iterations')
        plt.ylabel('cost')
        plt.title('cost vs #iterations with alpha of %e' % alp)
        '''

        return new_Z, new_para


if __name__ == '__main__':

    # train the model
    c = np.float_power(10, -np.arange(0, 10, 1))

    alps = []
    for i in range(len(c)):
        alps = alps + list(np.arange(0.1, 1.1, 0.1) * c[i])

    alps.sort(reverse=True)

    print(alps)

    new_Z = []
    new_para = []
    # i is for print the cost vs iteration figure
    for j in range(len(alphas)):
        para1 = []
        para2 = []
        for i, alp in enumerate(alps):
            c1, c2 = model(X, Y[j], alp, i + 2, learning_rate=0.003, training_epochs=5000)
            para1.append(c1)
            para2.append(c2)

        new_Z.append(para1)
        new_para.append(para2)


    Z2_test = []
    test_cost = []
    new_Z_test = []
    # test the model with trained parameters
    for j in range(len(alphas)):
        para3 = []
        para4 = []
        para5 = []
        para6 = []
        for i in range(len(alps)):
            X_phi, Y_phi = create_placeholders()
            c3 = forward_propation(X_phi, X_test, new_para[j][i])
            para3.append(c3)
            c4 = MSE_cost(c3, Y_phi)
            para4.append(c4)

            with tf.Session() as sess:
                c5 = sess.run(c3, feed_dict={X_phi: X_test, Y_phi: Y_test})
                para5.append(c5)
                c6 = sess.run(c4, feed_dict={X_phi: X_test, Y_phi: Y_test})
                para6.append(c6)

        print(np.shape(para6))

        Z2_test.append(para3)
        new_Z_test.append(para5)
        test_cost.append(para6)


    for j in range(len(alphas)):
        for i in range(len(alps)):
            print(('alpha is %3f, mixing rate at %3f :' % (alps[i], alphas[j])), test_cost[j][i])


    plt.figure(2)
    for j in range(len(alphas)):
        plt.semilogx(alps, test_cost[j], label='when mixing rate is %3f' % alphas[j])
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Test Error')
    plt.title(r'test Error vs $\alpha$ at all mixing rate')

    plt.show()


    """
    plt.figure(3)
    plt.plot(x_test.reshape(len(x_test), 1), Y_test.reshape(len(x_test), 1), 'bo', label='ground truth')
    
    for i in range(len(alps)):
        plt.plot(x_test.reshape(len(x_test), 1), new_Z_test[i].reshape(len(x_test), 1), label=('alpha = %e, test cost = %3f' % (alps[i], test_cost[i])))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('test Result with mixing rate of %f' % alpha)
    
    plt.figure(len(alps)+3)
    plt.plot(x1.reshape(len(x1), 1), Y.reshape(len(Y), 1), 'bo', label='ground truth')
    for i in range(len(alps)):
        plt.plot(x1.reshape(len(x1), 1), new_Z[i].reshape(len(x1), 1), label='alpha = %e' % alps[i])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression Result with mixing rate of %f' % alpha)
    """


    '''fig = plt.figure(5)
    ax = plt.axes(projection='3d')
    
    print(X_test[:1, :].shape)
    print(X_test[1:2, :].shape)
    print(Y_test.shape)
    print(new_Z_test.shape)
    
    axisx, axisy = np.meshgrid(X_test[:1, :], X_test[1:2, :])
    print(axisx.shape)
    print(axisy.shape)
    
    ax.scatter3D(X_test[:1, :], X_test[1:2, :], Y_test, color='red')
    ax.scatter3D(X_test[:1, :], X_test[1:2, :], new_Z_test, color='blue')
    plt.show()
    sys.exit(0)'''
