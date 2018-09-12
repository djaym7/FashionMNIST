# FashionMNIST
A CNN classifier for classifying Fashion MNIST database achieving over 96% accuracy.
Dataset can be downloaded as described in the iPython notebook.

FOLDERS:
Tensorboard Screenshots - contains screenshots of accuracy, loss, validation accuracy graphs
Logs - contains tensorboard logs


Here are the models configarations that performed the best:
"{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

Iter 1:
	least loss : 
			1-128-1, 
			2-128-1
			
	Most accurate:
			1-128-1, 
			2-128-1

	Most validation accuracy:
			2-128-1 by a good margin, 
			1-128-1
Iter 2:
	increased dense layers to 512, resulting in accuracy increasing to 98% but the training time increased from avg of 
	35s to avg of >80s per epoch.
	The least validation_accuracy was of 2-512-2 configaration 0.9124, having accuracy = 0.9720, loss=0.06399,val_loss=0.3881


		
