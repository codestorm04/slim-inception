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
import time
from datetime import datetime
import sklearn
import sklearn.metrics as metrics

import math
import tensorflow as tf

from datasets import dataset_factory, dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_integer(
    'batch_size', 5, 'The number of samples in each batch.')

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

FLAGS = tf.app.flags.FLAGS
acc_dic = {610: [0.050000000000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1860: [0.050000000000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8102: [0.98999999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.77272727272727271, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 6239: [0.97399999999999998, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.63636363636363635, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 6859: [0.98199999999999998, 1.0, 1.0, 1.0, 1.0, 1.0, 0.80000000000000004, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.68181818181818177, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3116: [0.33800000000000002, 0.0, 1.0, 0.76000000000000001, 0.0, 0.1111111111111111, 0.0, 0.6875, 0.0, 0.0, 0.95999999999999996, 0.73333333333333328, 0.10000000000000001, 0.13636363636363635, 0.32000000000000001, 0.41666666666666669, 0.61111111111111116, 0.045454545454545456, 0.17391304347826086, 0.31034482758620691, 0.0, 0.14285714285714285], 4366: [0.85999999999999999, 0.95454545454545459, 1.0, 0.95999999999999996, 0.81481481481481477, 1.0, 0.10000000000000001, 0.8125, 1.0, 0.33333333333333331, 0.95999999999999996, 0.83333333333333337, 1.0, 0.5, 0.95999999999999996, 0.94444444444444442, 0.97222222222222221, 0.86363636363636365, 0.86956521739130432, 1.0, 0.875, 0.7857142857142857], 1232: [0.050000000000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5617: [0.96999999999999997, 1.0, 1.0, 1.0, 1.0, 1.0, 0.40000000000000002, 0.9375, 1.0, 1.0, 1.0, 1.0, 1.0, 0.65217391304347827, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 8724: [0.996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.90909090909090906, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 7480: [0.98799999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72727272727272729, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2491: [0.13200000000000001, 0.0, 0.72727272727272729, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1111111111111111, 0.055555555555555552, 0.0, 0.30434782608695654, 0.27586206896551724, 0.0, 0.0], 3740: [0.63600000000000001, 0.59090909090909094, 1.0, 0.92000000000000004, 0.29629629629629628, 1.0, 0.0, 0.625, 0.73333333333333328, 0.083333333333333329, 0.88, 0.83333333333333337, 0.75, 0.31818181818181818, 0.80000000000000004, 0.94444444444444442, 0.80555555555555558, 0.22727272727272727, 0.56521739130434778, 0.75862068965517238, 0.0, 0.35714285714285715], 4991: [0.94999999999999996, 1.0, 1.0, 1.0, 1.0, 1.0, 0.20000000000000001, 0.875, 1.0, 0.95833333333333337, 0.95999999999999996, 0.93333333333333335, 1.0, 0.59090909090909094, 1.0, 1.0, 1.0, 1.0, 0.91304347826086951, 1.0, 1.0, 1.0]}

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()


    saver = tf.train.Saver(variables_to_restore)

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    probs_op = tf.nn.softmax(logits)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    # else:
    #   checkpoint_path = FLAGS.checkpoint_path

    checkpoints = glob.glob(FLAGS.checkpoint_path+'model*index')
    checkpoints.sort()
    for checkpoint in checkpoints[1:]:#
        checkpoint_path = checkpoint[:-6]

        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        if global_step in acc_dic.keys():
            continue
        print('Succesfully loaded model from %s at step=%d.' %
              (checkpoint_path, global_step))

        tf.logging.info('Evaluating %s' % checkpoint_path)
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=config) as sess:
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())

            saver.restore(sess, checkpoint_path)

            # csvPath = "predict/trainset_predction_"+'V4_focal_loss_5'+'_17574'+".csv"
            # file_csv = open(csvPath, 'w')
            # file_csv.write('xxxxxx,'+str(0.0000) + "," + str(int(0)) + "\n")

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                final_confusion_matrix = np.zeros((dataset.num_classes, dataset.num_classes))
                # print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
                start_time = time.time()
                i = 0
                step = 0
                while step < num_batches and not coord.should_stop():
                    gts, probs, preds = sess.run([labels, probs_op, predictions])
                    confusion_matrix = metrics.confusion_matrix(gts, preds, labels=range(dataset.num_classes))
                    final_confusion_matrix += confusion_matrix
                    step += 1

                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = FLAGS.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                              'sec/batch)' % (datetime.now(), step, num_batches,
                                              examples_per_sec, sec_per_batch))
                        start_time = time.time()
                    #     for imagename, prob, gt in zip(image_names, probs[:, 1:], gts):
                #         file_csv.write(imagename + "," + str(round(prob[0], 4)) + "," + str(gt) + "\n")
                #         i += 1
                #         if i % 100 == 0:
                #             print(i, ' examples!!')

                print('confusion_matrix:')
                print(final_confusion_matrix.astype(np.int32))
                accuracy_dic = {}
                t_count = 0
                labels_to_class_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
                each_class_total_count = np.sum(final_confusion_matrix, axis=1)
                total_sample_count = np.sum(final_confusion_matrix)
                for label_index in range(dataset.num_classes):
                    if each_class_total_count[label_index] == 0:
                        acc = 0.0
                    else:
                        acc = final_confusion_matrix[label_index][label_index]/each_class_total_count[label_index]
                    accuracy_dic[labels_to_class_names[label_index]] = (acc, each_class_total_count[label_index])
                    t_count += final_confusion_matrix[label_index][label_index]
                accuracy = t_count/total_sample_count


                acc_dic[global_step] = [accuracy]
                print('%s: accuracy = %.4f [%d examples]' %
                      (datetime.now(), accuracy, total_sample_count))
                # tumor_accuracy = total_true_positive_count / (total_true_positive_count+total_false_negative_count)
                # normal_accuracy = total_true_negative_count / (total_true_negative_count+total_false_positive_count)
                print('%s: accuracy of each classes:' % (datetime.now()))
                for label_index in range(dataset.num_classes):
                    label_name = labels_to_class_names[label_index]
                    acc_dic[global_step].append(accuracy_dic[label_name][0])
                    print('                            %s: accuracy = %.4f [%d examples]' %
                        (label_name, accuracy_dic[label_name][0], accuracy_dic[label_name][1]))
                # file_csv.close()

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=100)

    print(acc_dic)
    acc_sorted = sorted(acc_dic.items(), key = lambda x:x[1], reverse = True)
    print('sorted', acc_sorted)
    # draw_acc(acc_dic, '4_classes_511_4w_12w_sixth_filtered_SE_feature_fusion')


def draw_acc(accuracy_dic, model_name):
    accuracy = []
    normal_accuracy = []
    benign_accuracy = []
    insitu_accuracy = []
    invasive_accuracy = []
    steps = accuracy_dic.keys()
    steps.sort()
    print('steps:', steps)
    for step in steps:
        accuracy.append(accuracy_dic[step][0])
        normal_accuracy.append(accuracy_dic[step][1])
        benign_accuracy.append(accuracy_dic[step][2])
        insitu_accuracy.append(accuracy_dic[step][3])
        invasive_accuracy.append(accuracy_dic[step][4])

    # plt.figure(0, figsize=(15,15)).clf()
    plt.figure().clf()
    # plt.title(model_name)
    # r'$\alpha_i > \beta_i$'

    # plt.subplot(3,2,1)
    # r'$\alpha_i > \beta_i$'
    plt.scatter(steps, accuracy, c='b', marker='*', label='total')
    # plt.legend('accuracy', loc='upper center')
    plt.scatter(steps, normal_accuracy, c='g', marker='x', label='normal')
    # plt.legend('normal accuracy', loc='upper center')
    plt.scatter(steps, benign_accuracy, c='r', marker='x', label='benign')
    # plt.legend('benign accuracy', loc='upper center')
    plt.scatter(steps, insitu_accuracy, c='y', marker='x', label='insitu')
    plt.scatter(steps, invasive_accuracy, c='k', marker='x', label='invasive')
    # plt.legend('invasive accuracy', loc='upper center')
    plt.ylim([0, 1])
    plt.legend(ncol=2, loc='lower center')

    plt.title('Accuracy')# on 30000 test samples
    plt.grid(True, linestyle='-')

    plt.savefig("accuracy/"+model_name)


        # slim.evaluation.evaluate_once(
        #     master=FLAGS.master,
        #     checkpoint_path=checkpoint_path,
        #     logdir=FLAGS.eval_dir,
        #     num_evals=num_batches,
        #     eval_op=list(names_to_updates.values()),
        #     variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
