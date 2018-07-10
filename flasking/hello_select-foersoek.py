from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import re

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


def sample(prime, text_length):
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
            sam = model.sample(sess, chars, vocab, text_length, prime,
                               sample)
    tf.reset_default_graph()
    sampled_text = re.sub('\n','<br />',sam)
    return sampled_text







@app.route('/')
def home():
    return render_template('index.html')




@app.route('/SOU/<prime>/<int:text_length>')
def hello_world(prime, text_length):
    path = 'save'
    # list of all content in a directory, filtered so only directories are returned
    dir_list = [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path,directory))]
    print(dir_list)
    text = "Hej" #sample(prime, text_length)
    option_list = ""
    for o in dir_list:
        option_list = option_list + '<option value="'+o+'">'+o+'</option>'
    template = '''
<html>
    <head>
        <title>En alternativ SOU. Kim Svensson, AI</title>
    </head>
    <body width='500'><div>
        <h3>En alternativ SOU</h3>
        <p>'''+render_template('echo.html', value = option_list)+'''</p>
        <p><select>'''+option_list+ '''</select></p>
        <p>'''+text+ '''</p>
        </div>
    </body>
</html>'''
    return template

@app.route("/temp")
def hello():
    return render_template('echo.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
