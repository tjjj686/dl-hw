from uwnet import *
##def conv_net():
##    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
##            make_activation_layer(RELU),
##            make_maxpool_layer(16, 16, 8, 3, 2),
##            make_convolutional_layer(8, 8, 8, 16, 3, 1),
##            make_activation_layer(RELU),
##            make_maxpool_layer(8, 8, 16, 3, 2),
##            make_convolutional_layer(4, 4, 16, 32, 3, 1),
##            make_activation_layer(RELU),
##            make_connected_layer(512, 10),
##            make_activation_layer(SOFTMAX)]
##    return make_net(l)
## Without batchnorm(learning rate=0.01):
## ('training accuracy: %f', 0.3983199894428253)
## ('test accuracy:     %f', 0.4000000059604645)

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_batchnorm_layer(1),
            make_activation_layer(SOFTMAX)]
    return make_net(l)
##Apply batchnorm (learning rate=0.01):
##('training accuracy: %f', 0.5479599833488464)
##('test accuracy:     %f', 0.5418000221252441)

##Therefore, when apply batchnorm,the acuuracy increases a lot which is better.

##Apply batchnorm (learning rate=0.1):
##('training accuracy: %f', 0.5519800186157227)
##('test accuracy:     %f', 0.5472000241279602)

##Apply batchnorm (learning rate=0.001):
##('training accuracy: %f', 0.455159991979599)
##('test accuracy:     %f', 0.4586000144481659)



print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .001
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
## As we can see the best perfomance occurs when learning rate is 0.1, its test
## accuracy can achieve 54.7% using the default model.When decrease the learning
## rate, the performance drops.

