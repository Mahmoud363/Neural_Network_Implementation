from Layers import *

class Network(object):
    def __init__(self, network, prefix='best/'):
        self.prefix = prefix
        self.best_val =-1
        self.network = network
        # give names to each layer for saving
        for i in range(len(network)):
            network[i].name ='L'+str(i+1)
            network[i].prefix = prefix

    def forward(self,X, training = True):
          activations = []
          input = X
          # do the forward propagation
          for layer in self.network:
              #print('input dim: '+str(input.shape))
              activations.append(layer.forward(input, training = training))
              input=activations[-1]
          assert len(activations) == len(self.network)
          return activations

    def _regularize(self, X):
        reg_loss = 0
        # compute regularization loss
        for layer in self.network:
            reg_loss += layer.regularize_forward()
        return reg_loss     
    
    def train(self, X, Y):
        # get the losses
        layerAct = self.forward( X)
        reg_loss = self._regularize(X)
        
        layerIn = [X]+layerAct
        logit = layerAct[-1]
        # apply softmax function
        L = SoftMax(logit, Y)
        loss = L.forward()+reg_loss
        # do the backward propagation step
        lossG=L.backward()
        for ind in range(len(self.network))[::-1]:
            layer=self.network[ind]
            # each layer takes its previous input and the gradient so far
            # to compute its gradients
            lossG=layer.backward(layerIn[ind], lossG)
        return cp.mean(loss) # returnt the average loss
    
    def loss(self, X, Y):
        # a function the return the total loss to save it 
        # for the validation and training sets
        layerAct = self.forward( X)
        reg_loss = self._regularize(X)
        logit = layerAct[-1]
        L = SoftMax(logit, Y)
        loss = L.forward()+reg_loss
        return cp.mean(loss)


    def save_best(self, bestv):
        # save the current best weights
        self.best_val = bestv
        for layer in self.network:
            layer.save_best()
    
    def load_best(self):
        #load the current best weights
        for layer in self.network:
            layer.load_best()
    
    def save(self):
        #save all the network's valid layer weights
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)
        for layer in self.network:
            layer.save()
    
    def load(self):
        # load the weights
        for layer in self.network:
            layer.load()
    
    def fit(self, X):
        # pridict on X its classes
        loss = self.forward(X, training=False)[-1]
        return loss.argmax(axis=-1)
