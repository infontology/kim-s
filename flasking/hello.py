from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os

#from __future__ import print_function
import tensorflow as tf

import argparse
from six.moves import cPickle

from model import Model

from six import text_type


# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))






n = 5000
sample = 0


def sample(prime):
    with open('save/config.pkl', 'rb') as f:
        saved_args = cPickle.load(f)
    with open('save/chars_vocab.pkl', 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sam = model.sample(sess, chars, vocab, n, prime,
                               sample).encode('utf-8')
            print (sam.decode('utf-8'))
            with open("sample.txt", 'w') as f:
                f.write(str(sam.decode('utf-8')))
    tf.reset_default_graph()
    return sam.decode('utf-8')







@app.route('/')
def home():
    return render_template('index.html')



@app.route('/numbers/<user_name>/<prime>')
def hello_world(user_name, prime):
    text = sample(prime)
    return 'Hello, World! %s %s' % (user_name, text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
