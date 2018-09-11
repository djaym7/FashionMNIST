# FashionMNIST
A CNN classifier for classifying Fashion MNIST database achieving over 96% accuracy.
Dataset can be downloaded as described in the iPython notebook.

FOLDERS:
Tensorboard Screenshots - contains screenshots of accuracy, loss, validation accuracy graphs
Logs - contains tensorboard logs


Here are the models configarations that performed the best:
"{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

least loss : 
		1-128-1
		2-128-1
		
Most accurate:
		1-128-1
		2-128-1

Most validation accuracy:
		2-128-1 by a good margin
		1-128-1

least validation loss:
		2-32-2
		
