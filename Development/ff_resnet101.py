# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:48:43 2016

@author: Kevin Liang

Feed-forward ResNet-101 convolutional model: 
https://arxiv.org/pdf/1512.03385v1.pdf
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model
from Lib.TensorBase.tensorbase.base import Layers
from Lib.TensorBase.tensorbase.data import Mnist

import tensorflow as tf
import numpy as np
#import scipy.misc


# Global Dictionary of Flags
flags = {
    'data_directory': '../Data/MNIST/',
    'save_directory': '../Logs/summaries/',
    'model_directory': 'resnet101/',
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 10,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-7,
    'lr_decay': 0.999,
    'lr_iters': [(5e-3, 5000), (5e-3, 7500), (5e-4, 10000), (5e-5, 10000)]
}

class resnet101(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.data = Mnist(flags_input)
        
    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')

    def _set_summaries(self):
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)
        tf.summary.image("x", self.x)
        
    def _conv_layers(self,x):
        conv_layers = Layers(x)
        
        # Convolutional layers
        res_blocks = [1,3,4,23,3]
        output_channels = [64,256,512,1024,2048]
        
        with tf.variable_scope('scale0'):
            conv_layers.conv2d(filter_size=7,output_channels=output_channels[0],stride=2,padding='SAME',b_value=None)
            conv_layers.maxpool(k=3)
        with tf.variable_scope('scale1'):
            conv_layers.res_layer(filter_size=3, output_channels=output_channels[1], stride=2)
            for block in range(res_blocks[1]-1):
                conv_layers.conv_layers.res_layer(filter_size=3, output_channels=output_channels[1], stride=1)
        with tf.variable_scope('scale2'):
            conv_layers.res_layer(filter_size=3, output_channels=output_channels[2], stride=2)
            for block in range(res_blocks[2]-1):
                conv_layers.conv_layers.res_layer(filter_size=3, output_channels=output_channels[2], stride=1)
        with tf.variable_scope('scale3'):
            conv_layers.res_layer(filter_size=3, output_channels=output_channels[3], stride=2)
            for block in range(res_blocks[3]-1):
                conv_layers.conv_layers.res_layer(filter_size=3, output_channels=output_channels[3], stride=1)
        with tf.variable_scope('scale4'):
            conv_layers.res_layer(filter_size=3, output_channels=output_channels[4], stride=2)
            for block in range(res_blocks[4]-1):
                conv_layers.conv_layers.res_layer(filter_size=3, output_channels=output_channels[4], stride=1)
        
        conv_layers.avgpool(globe=True)
        
        # Fully Connected Layer
        conv_layers.fc(output_nodes=10)

        return conv_layers.get_output()
            
    def _network(self):
        with tf.variable_scope("conv_layers"):
            self.pyx = self._conv_layers(x=self.x)
            self.logits = tf.nn.softmax(self.pyx)

    def _optimizer(self):
        const = 1/self.flags['batch_size']
        self.xentropy = const * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.pyx, labels=self.y, name='xentropy'))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        
    def _generate_train_batch(self):
        self.train_batch_y, train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.train_batch_x = np.reshape(train_batch_x, [self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])

    def _generate_valid_batch(self):
        self.valid_batch_y, valid_batch_x, valid_number, batch_size = self.data.next_valid_batch(self.flags['batch_size'])
        self.valid_batch_x = np.reshape(valid_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])
        return valid_number

    def _generate_test_batch(self):
        self.test_batch_y, test_batch_x, test_number, batch_size = self.data.next_test_batch(self.flags['batch_size'])
        self.test_batch_x = np.reshape(test_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])
        return test_number

    def _run_train_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                                                   self.lr: rate})

    def _run_train_summary_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, self.loss, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                   feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                                                              self.lr: rate})
                                                              
    def _run_valid_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.valid_batch_x})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.valid_batch_y, 1), np.argmax(predictions, 1))
        self.valid_results = np.concatenate((self.valid_results, correct_prediction))

    def _run_test_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.test_batch_x})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.test_batch_y, 1), np.argmax(predictions, 1))
        self.test_results = np.concatenate((self.test_results, correct_prediction))

    def _record_train_metrics(self):
        self.print_log("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(self.loss))

    def _record_valid_metrics(self):
        accuracy = np.mean(self.valid_results)
        self.print_log("Accuracy on Validation Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'ValidAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()

    def _record_test_metrics(self):
        accuracy = np.mean(self.test_results)
        self.print_log("Accuracy on Test Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'TestAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_resnet101 = resnet101(flags, run_num=1)
    model_resnet101.train()


if __name__ == "__main__":
    main()