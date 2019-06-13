# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")
import importlib
importlib.reload(sys)

import os
import glob
import numpy as np
import numpy.random as random
import time
from datetime import datetime
import sklearn
import sklearn.metrics as metrics

import math
import tensorflow as tf

from datasets import dataset_factory, dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory
import cv2
from PIL import Image, ImageDraw, ImageFont
slim = tf.contrib.slim
import process_img_lyz


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/sde1/products/inception_v3_training_422/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/sde1/products/inception_v3_eval/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'products', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/sde1/products', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_integer(
    'num_classes', 22, 'output of the network predictions')

tf.app.flags.DEFINE_float(
    'threshhold', 0.25, 'the threshhold of predicting a class of goods')

FLAGS = tf.app.flags.FLAGS

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():


        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)


        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        class_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
        image_x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image = image_preprocessing_fn(image_x, eval_image_size, eval_image_size)

        images = tf.expand_dims(image, 0)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        predictions = tf.argmax(logits, 1)
        probs_op = tf.nn.softmax(logits)

        checkpoints = glob.glob(FLAGS.checkpoint_path+'model*index')
        checkpoints.sort()
        checkpoint_path = checkpoints[-1][:-6]
        tf.logging.info('Evaluating %s' % checkpoint_path)

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            saver.restore(sess, checkpoint_path)
            print('Succesfully loaded model from %s.' % checkpoint_path)

            start_time = time.time()
            i = 0
            step = 0
            fps = 0
            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture('../1.mp4')

            while True: 
                # path = input("path of picture: ")
                # frame = cv2.imread('../' + path)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ret, frame = cap.read()
                input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_img, x, y, w, h = process_img_lyz.bounding_crop(input_img, eval_image_size, eval_image_size)

                if ret == True and frame.size != 0:
                    if step % 5 == 0:
                        input_img = input_img.astype(np.float32)  
                        input_img = input_img / 255.0

                        probs, preds = sess.run([probs_op, predictions], {image_x: input_img})
                        pred = preds[0]
                        label = class_names[pred]
                        prob = round(probs[0][pred] * 100, 4)
                        print(str(pred) + ' -- ' + label + ' --------- ' + str(prob) + '%')

                        duration = time.time() - start_time
                        fps = 5.0 / duration 
                        print("processing images at %f fps" % fps)
                        start_time = time.time()

                    step += 1
                    if prob > FLAGS.threshhold * 100 and class_names[pred] != '0_bg':
                        frame = _put_chinese(frame, str(pred) + ' -- ' + label, (10, 0), (30, 255,30))
                        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        frame = _put_chinese(frame, '未检测到商品', (10, 0), (0, 0,255))
                    cv2.putText(frame, 'Confidence: ' + str(prob) + '%', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, 1)
                    cv2.putText(frame, 'FPS: %f' % fps, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, 1)        
                    cv2.imshow('video', frame)
                # Use Esc key to close the program
                key = cv2.waitKey(5) & 0xff        
                if key == 27:
                    break

            #Realse & destroy
            cap.release()
            cv2.destroyAllWindows()

def _put_chinese(image, text, position, color):
    img_PIL = Image.fromarray(image) 
    # Use PIL without FreeType lib to print Chinese text 
    # 字体 字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc 
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 17) 
     
    # 需要先把输出的中文字符转换成Unicode编码形式 
    if not isinstance(text, str): 
        text = text.decode('utf8') 
      
    draw = ImageDraw.Draw(img_PIL) 
    draw.text(position, text, font=font, fill=color) 
      
    # 转换回OpenCV格式 
    frame = np.asarray(img_PIL) 
    return frame

if __name__ == '__main__':
  tf.app.run()


# python3 predict_lyz.py  --dataset_dir="/home/lyz/desktop/github_repos/1.7-code/goods_data_299x299_croped/" --checkpoint_path="/home/lyz/desktop/github_repos/1.7-code/goods_data_299x299_croped/training/"

# full image training and testing