# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:58:10 2017

@author: jens
"""

import numpy as np
import sc_network
import sc_config
import sc_reader_test

import tensorflow as tf

def run_data(dat,model):


    config = sc_config.ACConfig(model_name=model, is_training=False)

    hyp = run(dat,config)
    
    return hyp
    
def run(dat,config):
    
	
     with tf.Graph().as_default() as g:

        m = sc_network.SCModel(config)

        s = tf.train.Saver(tf.all_variables())
			
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
				
            ckpt = tf.train.get_checkpoint_state(config.model_dir_test)
            s.restore(session, ckpt.model_checkpoint_path)
            N_outputs = int(np.floor(dat.shape[1]/(4*15)))
            
            state = np.zeros([1,config.num_hidden*2])
            dat = np.expand_dims(dat.T,axis=0)
            dat = dat[:,:N_outputs*4*15]
					
            prediction, _ = session.run([m.logits, m.final_state], feed_dict={
                        m.features: dat,
                        m.targets: np.ones([N_outputs,5]),
                        m.mask: np.ones(N_outputs),
                        m.batch_size: np.ones([1]),
                        m.initial_state: state                        
                        })

            return prediction
