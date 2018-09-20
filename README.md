# FashionMNIST
A CNN classifier for classifying Fashion MNIST database achieving over 96% accuracy.
Dataset can be downloaded as described in the iPython notebook.

FOLDERS:
Tensorboard Screenshots - contains screenshots of accuracy, loss, validation accuracy graphs
Logs - contains tensorboard logs


Here are the models configarations that performed the best:
"{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

Iter 1: Tried various combinations of configs:

		dense_layers = [0,1,2]
		layer_sizes = [32,64,128] #dense layer sizes
		conv_layers = [1,2,3]
		
	Result:
	least loss : 
			1-128-1, 
			2-128-1
			
	Most accurate:
			1-128-1, 
			2-128-1

	Most validation accuracy:
			2-128-1 by a good margin, 
			1-128-1
			
Iter 2: Increased dense layers to 512

	Result:
		Accuracy increased to 98% 
		Training time increased from avg of 35s to avg of >80s per epoch.
	Best Configs:	
		2-512-2 configaration 
		accuracy = 0.9720, 
		loss=0.06399,
		val_loss=0.3881

RNN model: 

	To test it out, the the model was tested with the same dataset and achieved accuracy of over 85% but 
	took more resources and time compared to CNN model.
			
