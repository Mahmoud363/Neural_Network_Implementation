from Network import *

class Trainer(object):
    def __init__(self, model, data, batch_size = 1000, epochs = 50, verbose = True):

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.best_val = -1
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.verbose = verbose
        self.best_epoch = 0
        self.loss_history = []
        self.loss_history_val = []

        self.train_acc_history = []
        self.val_acc_history = []

    def _step(self, t):
        # do the forward and backward steps
        num_train = self.X_train.shape[0]
        #divide the data to get the current batch
        X_batch = self.X_train[t*self.batch_size: (t+1)*self.batch_size]
        y_batch = self.y_train[t*self.batch_size: (t+1)*self.batch_size]
        loss = self.model.train(X_batch, y_batch)
        self.loss_history.append(loss)

    def check_accuracy(self, X, y):
        #compute the current accuracy for the current weights
        N = X.shape[0]
        num_batches = N // self.batch_size
        if N % self.batch_size != 0:
            num_batches += 1
        y_pred = []
    
        scores = self.model.forward(X, training=False)[-1]
        y_pred.append(cp.argmax(scores, axis=1))
        y_pred = cp.hstack(y_pred)
        acc = cp.mean(y_pred == y)
        return acc

    def train(self):

        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)

        for epoch in range(self.num_epochs):
            for batch in range(iterations_per_epoch):
                self._step(batch) # train on the current batch
            #get training and validation accuracies
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            val_acc = self.check_accuracy(self.X_val, self.y_val)    
            # append the current values to history lists to generate the plots
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self.loss_history_val.append(self.model.loss(self.X_val, self.y_val))

            if self.verbose:
                print('(Epoch %d / %d) train_loss: %f; train acc: %f; val_loss:%f; val_acc: %f' % (
                           epoch+1, self.num_epochs, self.loss_history[-1], train_acc, self.loss_history_val[-1], val_acc))
            #save the best weights
            if val_acc >= self.best_val:
                self.best_val = val_acc
                self.best_epoch = epoch
                self.model.save_best(val_acc)
        # load the best weights based on validation weights
    
    def load_best(self):
        self.model.load_best()
    

