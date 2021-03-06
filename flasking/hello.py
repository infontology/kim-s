from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify, make_response
import atexit
import cf_deployment_tracker
import os
import re
import random
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import html


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
data=[{'dir':'save', 'name':'SOU'}, {'dir':'save/hammar_sou_livskvalitet_blandat_120000t/save', 'name':'SOU+Hammar'}, {'dir':'save/hammar/save', 'name':'Hammar'}]


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

def sample_select(prime, text_length, dir):
    with open(dir +'/config.pkl', 'rb') as f:
        saved_args = cPickle.load(f)
    with open(dir +'/chars_vocab.pkl', 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sam = model.sample(sess, chars, vocab, text_length, prime,
                               sample)
    tf.reset_default_graph()
    sampled_text = sam # re.sub('\n','<br />',sam)
    return sampled_text





@app.route('/')
def home():
    return render_template('index.html')



# Härifrån: https://gist.github.com/rduplain/1641344 Ändrat till io enligt en kommentar
@app.route('/plot/')
def plot():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]

    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

# Titta här! https://stackoverflow.com/questions/32019733/getting-value-from-select-tag-using-flask

@app.route('/select/')
def select():
    return render_template(
        'select.html',
        data=data)

@app.route('/text/' , methods=['GET', 'POST'])
def generate_text():
    prime = 'Utredningen'
    text_length = 2000
    select = request.form.get('comp_select')
    prime = request.form.get('prime')
    text_length = int(request.form.get('length'))
    text = sample_select(prime, text_length, select)
    textmaengd = next(item for item in data if item["dir"] == select)['name']
    title = "Text baserat på " + prime + ' från ' + textmaengd
    text = html.unescape(text)
    print(text)
    return render_template('text.html', data=text, title=title)


@app.route('/SOU/<prime>/<int:text_length>')
def hello_world2(prime, text_length):
    text = sample(prime, text_length)
    template = '''
<html>
    <head>
        <title>En alternativ SOU. Kim Svensson, AI</title>
    </head>
    <body width='500'><div>
        <h3>En alternativ SOU</h3>
        <p>'''+text+ '''</p>
        </div>
    </body>
</html>'''
    return template


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
