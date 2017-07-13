import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 60, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

