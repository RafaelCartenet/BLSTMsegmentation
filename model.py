
# Libraries
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM
import tensorflow as tf
import numpy as np
import time
import sys
import os


class GroupClassifier:
    def __init__(self,
                 models_path,   # path where will be save/restore models
                 model_name,    # name of the folder of models
                 DEBUG=0,       # if 1, uses mini dataset
                 VERBOSE=1,     # if 1, prints additionnal informations
                 SAVE=1):       # if 1, save model after each epoch
        self.DEBUG= DEBUG
        self.VERBOSE= VERBOSE
        self.SAVE= SAVE
        self.MODELS_PATH= models_path
        self.MODEL_NAME= model_name
        self.logger= Logger(VERBOSE)

        self.max_length= 800

    def load_data(self, X_train, Y_train, X_test, Y_test):
        self.X_train= np.asarray(X_train)
        self.Y_train= np.asarray(Y_train)
        self.X_test= np.asarray(X_test)
        self.Y_test= np.asarray(Y_test)

        # Determine the set of labels we are working on
        labels= set(np.unique(self.Y_train))
        labels= labels.union(set(np.unique(self.Y_test)))
        self.labels= sorted(labels)

        # Define the input and output size
        self.input_size= self.X_train[0].shape[1]
        self.output_size= len(self.labels)

    def arrange_data(self, batchsize):
        logstring=  '\nDATA SHAPES BEFORE ARRANGEMENT\n'
        logstring+= 'X_train shape %s\n'% (self.X_train.shape,)
        logstring+= 'Y_train shape %s\n'% (self.Y_train.shape,)
        logstring+= 'X_test shape %s\n'% (self.X_test.shape,)
        logstring+= 'Y_test shape %s\n'% (self.Y_test.shape,)
        self.logger.write_log(logstring)

        nb_trainsamples= len(self.X_train)
        nb_testsamples= len(self.X_test)
        self.batchsize= batchsize

        # TRANSFORM LABELS TO ONE HOT VECTORS.
        # Training data labels
        onehots= np.zeros((nb_trainsamples, self.max_length, self.output_size), dtype='f')
        for i in range(nb_trainsamples):
            for j in range(len(self.Y_train[i])):
                ind= self.labels.index(self.Y_train[i][j])
                onehots[i, j, ind]= 1
        self.Y_train= onehots

        # Testing data labels
        onehots= np.zeros((nb_testsamples, self.max_length, self.output_size), dtype='f')
        for i in range(nb_testsamples):
            for j in range(len(self.Y_test[i])):
                ind= self.labels.index(self.Y_test[i][j])
                onehots[i, j, ind]= 1
        self.Y_test= onehots

        # # Compute maximum length among samples
        # max_length= max([len(sample) for sample in np.concatenate((self.X_train, self.X_test), axis=0)])


        # Create np arrays for X and lengths
        X_train= np.zeros((nb_trainsamples, self.max_length, self.input_size), dtype='f')
        self.lengths_train= np.zeros((nb_trainsamples), dtype=np.uint32)
        X_test= np.zeros((nb_testsamples, self.max_length, self.input_size), dtype='f')
        self.lengths_test= np.zeros((nb_testsamples), dtype=np.uint32)

        # Fill the created arrays with the data
        # training data
        for i in range(nb_trainsamples):
            length= int(len(self.X_train[i]))
            self.lengths_train[i]= length
            X_train[i,:length]= self.X_train[i]
        self.X_train= X_train
        # testing data
        for i in range(nb_testsamples):
            length= int(len(self.X_test[i]))
            self.lengths_test[i]= length
            X_test[i,:length]= self.X_test[i]
        self.X_test= X_test

        # Compute number of batches for each set
        self.nb_batch_train= self.X_train.shape[0]//self.batchsize
        self.nb_batch_test= self.X_test.shape[0]//self.batchsize

        # Shuffle the training set according to a random permutation
        train_perm= np.random.permutation(self.nb_batch_train*self.batchsize)
        self.X_train= self.X_train[:self.nb_batch_train*self.batchsize][train_perm]
        self.Y_train= self.Y_train[:self.nb_batch_train*self.batchsize][train_perm]
        self.lengths_train= self.lengths_train[:self.nb_batch_train*self.batchsize][train_perm]

        # Shuffle the testing set according to a random permutation
        test_perm= np.random.permutation(self.nb_batch_test*self.batchsize)
        self.X_test= self.X_test[:self.nb_batch_test*self.batchsize][test_perm]
        self.Y_test= self.Y_test[:self.nb_batch_test*self.batchsize][test_perm]
        self.lengths_test= self.lengths_test[:self.nb_batch_test*self.batchsize][test_perm]

        # Reshape training data to batches
        self.X_train= self.X_train.reshape(self.nb_batch_train, -1, self.max_length, self.input_size)
        self.Y_train= self.Y_train.reshape(self.nb_batch_train, -1, self.max_length, self.output_size)
        self.lengths_train= self.lengths_train.reshape(self.nb_batch_train, -1)

        # Reshape testing data to batches
        self.X_test= self.X_test.reshape(self.nb_batch_test, -1, self.max_length, self.input_size)
        self.Y_test= self.Y_test.reshape(self.nb_batch_test, -1, self.max_length, self.output_size)
        self.lengths_test= self.lengths_test.reshape(self.nb_batch_test, -1)

        logstring=  '\nDATA SHAPES AFTER ARRANGEMENT\n'
        logstring+= 'X_train shape %s\n'% (self.X_train.shape,)
        logstring+= 'Y_train shape %s\n'% (self.Y_train.shape,)
        logstring+= 'X_test shape %s\n'% (self.X_test.shape,)
        logstring+= 'Y_test shape %s\n'% (self.Y_test.shape,)
        logstring+= 'lengths_train %s\n'% (self.lengths_train.shape,)
        logstring+= 'lengths_test %s\n'% (self.lengths_test.shape,)
        self.logger.write_log(logstring)


    def extract_lasts(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        return tf.gather_nd(data, indices)

    #### Building Graph ####
    def build_graph(self,
            nb_hiddenunits=100,
            learning_rate=1e-4,
            name="",
            dropout=-1,
            dropoutwrapper=-1,
            FCL_size=30
        ):
        self.build_graph_variables(
            nb_hiddenunits=nb_hiddenunits,
            fully_connected_layer_size=FCL_size,
            dropout=dropout,
            dropoutwrapper=dropoutwrapper
        )
        self.build_graph_optimizer(learning_rate)
        self.build_graph_summaries(name)

        logstring=  '\nGRAPH VARIABLES SHAPES:\n'
        logstring+= 'INPUT:\t\t%s\n'% (self.X_.get_shape())
        logstring+= 'OUTPUT:\t\t%s\n'% (self.Y_.get_shape())
        logstring+= 'RNN outputs:\t%s\n'% (self.RNNoutputs.get_shape())
        logstring+= 'W_1 shape:\t%s\n'% (self.W_1.get_shape())
        logstring+= 'b_1 shape:\t%s\n'% (self.b_1.get_shape())
        logstring+= 'W_f shape:\t%s\n'% (self.W_f.get_shape())
        logstring+= 'b_f shape:\t%s\n'% (self.b_f.get_shape())
        logstring+= 'Predictions:\t%s\n'% (self.predictions.get_shape())
        self.logger.write_log(logstring)

    def build_graph_variables(self,
            nb_hiddenunits,
            fully_connected_layer_size,
            dropout,
            dropoutwrapper
        ):
        """ Implementation of Bidirectionnal LSTM for framewise classification
        of sequences of different lengths
        """

        tf.reset_default_graph()

        with tf.variable_scope("groups"):
            # Input placeholders
            self.X_= tf.placeholder(tf.float32, [None, None, self.input_size], name='input')
            self.Y_= tf.placeholder(tf.float32, [None, None, self.output_size], name='output')
            self.seq_lengths= tf.placeholder(tf.int32, [None], name='seq_length')
            self.keep_prob_inter= tf.placeholder_with_default(1., shape=(), name="keep_prob_inter")
            self.keep_prob_warp= tf.placeholder_with_default(1., shape=(), name="keep_prob_warp")

            # Bidirectionnal Reccurent Neural Network
            LSTM_fw_cell= tf.contrib.rnn.LSTMCell(nb_hiddenunits) # forward direction cell
            LSTM_bw_cell= tf.contrib.rnn.LSTMCell(nb_hiddenunits) # backward direction cell

            # add dropout warper
            LSTM_fw_cell= tf.contrib.rnn.DropoutWrapper(LSTM_fw_cell, output_keep_prob=self.keep_prob_warp)
            LSTM_bw_cell= tf.contrib.rnn.DropoutWrapper(LSTM_bw_cell, output_keep_prob=self.keep_prob_warp)

            outputs,_= tf.nn.bidirectional_dynamic_rnn(
                LSTM_fw_cell,
                LSTM_bw_cell,
                self.X_,
                sequence_length=self.seq_lengths,
                dtype=tf.float32
            )

            out_FW, out_BW= outputs
            self.RNNoutputs= tf.concat((out_FW, out_BW), 2)

            # Concat the outputs for every sample of batch for matrix multiplication
            self.outputs= tf.reshape(self.RNNoutputs, [-1, 2*nb_hiddenunits])

            # Inter Fully connected layer
            interFCLlength= fully_connected_layer_size
            self.W_1= tf.Variable(tf.truncated_normal([2*nb_hiddenunits, interFCLlength], stddev= 0.1), name= "FC1_weights")
            self.b_1= tf.Variable(tf.truncated_normal([interFCLlength], stddev=0.1), name= "FC1_bias")

            # Matrix multiplication
            self.outputs= tf.matmul(self.outputs, self.W_1) + self.b_1

            # Final fully connected layer
            self.W_f= tf.Variable(tf.truncated_normal([interFCLlength, self.output_size], stddev= 0.1), name= "FCL_weights")
            self.b_f= tf.Variable(tf.truncated_normal([self.output_size], stddev=0.1), name= "FCL_bias")

            # Softmax layer
            self.predictions= tf.nn.softmax(tf.matmul(self.outputs, self.W_f) + self.b_f)

            # Back to initial shape!
            batchsize= tf.shape(self.X_)[0]
            max_length= tf.shape(self.X_)[1]
            self.predictions= tf.reshape(self.predictions, [batchsize, max_length, self.output_size], name='predictions')

        # saver to store and restore variables
        self.saver= tf.train.Saver()

    def cost(self, output, target):
        # Compute cross entropy for each frame.
        cross_entropy= target * tf.log(output)
        cross_entropy= -tf.reduce_sum(cross_entropy, 2)

        # Average over actual sequence lengths.
        mask= tf.sign(tf.reduce_max(tf.abs(target), 2))
        cross_entropy*= mask
        cross_entropy= tf.reduce_sum(cross_entropy, 1)
        cross_entropy/= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(cross_entropy)

    def accuracy(self, output, target):
        # Compute accuracy for each frame.
        accuracy= tf.equal(tf.argmax(self.Y_, 2), tf.argmax(self.predictions, 2))
        accuracy= tf.cast(accuracy, tf.float32)

        # Average over actual sequence lengths.
        mask= tf.sign(tf.reduce_max(tf.abs(target), 2))
        accuracy*= mask
        accuracy= tf.reduce_sum(accuracy, 1)
        accuracy/= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(accuracy)

    def build_graph_optimizer(self, learning_rate=1e-4,):
        # Cross Entropy
        self.cross_entropy= self.cost(self.predictions, self.Y_)

        # Accuracy
        self.acc= self.accuracy(self.predictions, self.Y_)

        regularizer= tf.nn.l2_loss(self.W_f)
        self.loss= tf.reduce_mean(self.cross_entropy + .01*regularizer)

        # Optimizer
        self.learning_rate= learning_rate
        self.optimizer= tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def build_graph_summaries(self, name= ""):
        self.name= name
        tf.summary.scalar(self.name+'acc', self.acc)
        tf.summary.scalar(self.name+'loss', self.loss)
        self.summary_op = tf.summary.merge_all()


    def train_model(self,
            keep_prob_inter,
            keep_prob_warp,
            nb_epoch=200
        ):
        """ Train the model for nb_epoch iterations.
            Model is restored from models_path/model_name
            If model could not be loaded, random initialisation for parameters
            After each epoch the model parameters are saved and model
            is tested on the whole testing set
        """
        # Initialize the variables
        sess= tf.Session()
        sess.run(tf.global_variables_initializer())

        # Load variables if needed
        if self.SAVE:
            try:
                modelname= self.MODELS_PATH + self.name
                saver= tf.train.import_meta_graph(modelname + ".meta")
                self.saver.restore(sess, modelname)
            except:
                print "Failed to restore. Starting model from scratch."

        # Define log writers for Tensorboard
        self.file_writer_train= tf.summary.FileWriter("tflogs/training_set/"+self.name, sess.graph)
        self.file_writer_test= tf.summary.FileWriter("tflogs/testing_set/"+self.name, sess.graph)

        for epoch_id in range(nb_epoch):
            #### TRAINING ####
            epoch_time= time.time()
            epoch_acc= 0.
            epoch_loss= 0.
            print "epo#%s\t|batch#\t|accu\t|loss\t|time\t|"% (epoch_id+1)
            for batch_id in range(self.nb_batch_train):
                batch_time= time.time()

              # input
                batch_X, batch_Y= self.X_train[batch_id], self.Y_train[batch_id]
                lengths= self.lengths_train[batch_id]
              # optimize and compute loss + accuracy
                _, loss, acc, summary= sess.run(
                    fetches= [
                        self.optimizer,
                        self.loss,
                        self.acc,
                        self.summary_op],
                    feed_dict= {
                        self.X_: batch_X,
                        self.Y_: batch_Y,
                        self.seq_lengths: lengths,
                        self.keep_prob_inter: keep_prob_inter,
                        self.keep_prob_warp: keep_prob_warp
                    }
                )
              # update infos
                epoch_acc+= acc;epoch_loss+= loss
                batch_time= time.time()-batch_time

              # display results
                if batch_id % 30 == 0:
                    self.logger.write_log("\t|%s\t|%.2f%%\t|%.2f\t|%.2fs\t|"% (batch_id, 100*acc, loss, batch_time))

              # write log to tensorboard
                self.file_writer_train.add_summary(summary, epoch_id*self.nb_batch_train+ batch_id)

            # update epoch infos
            epoch_acc/= self.nb_batch_train
            epoch_loss/= self.nb_batch_train
            epoch_time= time.time()- epoch_time
            self.logger.write_log("\t|ALL\t|%.2f%%\t|%.2f\t|%.2fs\t|"% (100*epoch_acc, epoch_loss, epoch_time))


            #### SAVING MODEL ####
            if self.SAVE:
                self.saver.save(sess, self.MODELS_PATH + self.name)


            #### TESTING ####
            testing_time= time.time()
            testing_acc= 0
            testing_loss= 0
            for batch_id in range(self.nb_batch_test):

              # input
                batch_X, batch_Y= self.X_test[batch_id], self.Y_test[batch_id]
                lengths= self.lengths_test[batch_id]
              # compute loss + accuracy
                loss, acc, summary= sess.run(
                    fetches= [
                        self.loss,
                        self.acc,
                        self.summary_op
                    ],
                    feed_dict= {
                        self.X_: batch_X,
                        self.Y_: batch_Y,
                        self.seq_lengths: lengths
                    }
                )
              # update infos
                testing_acc+= acc
                testing_loss+= loss

              # write log to tensorboard
                self.file_writer_test.add_summary(summary, epoch_id*self.nb_batch_test+ batch_id)

            testing_time= time.time()- testing_time
            testing_acc/= self.nb_batch_test
            testing_loss/= self.nb_batch_test
            self.logger.write_log("\t|TEST\t|%.2f%%\t|%.2f\t|%.2fs\t|\n"% (100*testing_acc, testing_loss, testing_time))

    def evaluate_model(self):
        # Initialize the variables
        sess= tf.Session()
        sess.run(tf.global_variables_initializer())

        # Load variables
        if self.SAVE:
            try:
                modelname= self.MODELS_PATH + self.name
                saver= tf.train.import_meta_graph(modelname + ".meta")
                self.saver.restore(sess, modelname)
            except:
                print "Failed to restore model. Exiting."
                exit()

        #### TESTING ####
        Y_true= []
        Y_pred= []
        testing_time= time.time()
        testing_acc= 0
        testing_loss= 0
        tophonetic= np.vectorize(lambda t: sorted(self.labels)[t])
        for batch_id in range(self.nb_batch_test):
            batch_time= time.time()

          # input
            batch_X, batch_Y= self.X_test[batch_id], self.Y_test[batch_id]
            lengths= self.lengths_test[batch_id]


          # compute loss + accuracy
            loss, acc, predictions= sess.run(
                fetches= [
                    self.loss,
                    self.acc,
                    self.predictions
                ],
                feed_dict= {
                    self.X_: batch_X,
                    self.Y_: batch_Y,
                    self.seq_lengths: lengths
                }
            )
          # update infos
            testing_acc+= acc
            testing_loss+= loss

            for i in range(self.batchsize):
                true= batch_Y[i,:lengths[i]]
                true= np.argmax(true, axis=1)
                Y_true+= list(true)
                pred= predictions[i,:lengths[i]]
                pred= np.argmax(pred, axis=1)
                Y_pred+= list(pred)


        testing_time= time.time()- testing_time
        testing_acc/= self.nb_batch_test
        testing_loss/= self.nb_batch_test
        self.logger.write_log("\n\nAccuracy:\t%.2f%%\nLoss:\t\t%s\nTime:\t\t%.2fs\n"% (100*testing_acc, testing_loss, testing_time))

        Y_true= tophonetic(Y_true)
        Y_pred= tophonetic(Y_pred)

        self.logger.write_log(CR(Y_true, Y_pred))
        mat= CM(Y_true, Y_pred)
        groups= [group[:5] for group in sorted(self.labels)]
        CONFMAT= "\t" + "\t".join(groups) + "\n"
        for i,phonetic in enumerate(sorted(self.labels)):
            CONFMAT+= phonetic[:5] + "\t"+"\t".join(map(str, mat[i].tolist()+[np.sum(mat[i])])) + "\n\n"
        CONFMAT+= "\t" + "\t".join(map(str, np.sum(mat, axis=0).tolist()))
        self.logger.write_log(CONFMAT)


    def init_session(self):
        """ Method to inialize a new tensorflow session.
            From the model path and the model name given,
                - Load the meta graph
                - Restore saved variables
            No need to build graph from scratch.
            Returns the session so that it can be saved in cache.
        """
        # Initialize a new tensorflow session
        tf.reset_default_graph()
        session= tf.Session()
        session.run(tf.global_variables_initializer())

        # Define full path of the model
        full_path_model= self.MODELS_PATH + self.MODEL_NAME + "groups"

        try:
            # Restore graph from the .meta file
            saver= tf.train.import_meta_graph(full_path_model + ".meta")

            # Restore parameters
            saver.restore(session, full_path_model)
        except:
            print "Failed to restore BLSTM model: %s.\nEXIT."% (full_path_model)
            exit()
        return session


    def predict(self, session, data):
        """ Method to predict output from new samples, data.
            It is using a preloaded tensorflow session.
            Format of data should be [batchsize, X, self.inputsize]
        """
        # Determine batchsize and input size according to data
        batchsize= data.shape[0]
        max_length= data.shape[1]
        input_size= data[0][1].shape[0]

        # Load datastructures to store length and new data
        lengths= np.zeros((batchsize))
        new_data= np.zeros((batchsize, max_length, input_size), dtype='f')

        # Compute the length of each sample and apply zero padding
        for i in range(batchsize):
            lengths[i]= len(data[i])
            new_data[i,:int(lengths[i])]= data[i]

        # Compute the prediction using pre-loaded session
        predictions= session.run(
            fetches= [
                'groups/predictions:0'
            ],
            feed_dict= {
                'groups/input:0': new_data,
                'groups/seq_length:0': lengths
            }
        )

        return predictions


def forward_test():
    Clfr= GroupClassifier()
    Clfr.load_data()
    Clfr.arrange_data()
    Clfr.build_graph()

    sess= tf.Session()
    sess.run(tf.global_variables_initializer())

    X= Clfr.X_train[0]
    Y= Clfr.Y_train[0]
    lengths= Clfr.lengths_train[0]

    outputs, predictions, acc, loss= sess.run(
        [Clfr.X_, Clfr.predictions, Clfr.acc, Clfr.cross_entropy],
        feed_dict={
            Clfr.X_: X,
            Clfr.Y_: Y,
            Clfr.seq_lengths: lengths,
        }
    )

    print "\nPREDICTIONS\n", predictions
    print "\nOUTPUTS\n", outputs
    print "\nREAL LABELS\n", Y
    print "\nACCURACY\n%.2f%%"% acc
    print "\nLOSS\n%s"% loss

    print Y[0][50]
    print predictions[0][50]

if __name__ == "__main__":
    start= time.time()
    forward_test()
    assert()

    print 'Running time: %.2f' % (time.time() - start)
