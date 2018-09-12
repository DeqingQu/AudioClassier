import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt
# from PIL import Image
import time

def get_files(filepath, train_ratio=0.8):
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    foldernames = os.listdir(filepath)
    for folder in foldernames:
        directory = filepath + folder + "/result/specgram/"
        filenames = os.listdir(directory)
        num_train_files = len(filenames) * train_ratio
        for i, filename in enumerate(filenames):
            if i < num_train_files:
                train_files.append(directory + filename)
                label = filename[4]
                if label == 'z':
                    label = 0
                else:
                    label = int(label)
                train_labels.append(label)
            else:
                test_files.append(directory + filename)
                label = filename[4]
                if label == 'z':
                    label = 0
                else:
                    label = int(label)
                test_labels.append(label)

    train_image_list = np.hstack((train_files))
    train_label_list = np.hstack((train_labels))
    test_image_list = np.hstack((test_files))
    test_label_list = np.hstack((test_labels))

    train_temp = np.array([train_image_list, train_label_list])
    train_temp = train_temp.transpose()
    np.random.shuffle(train_temp)
    test_temp = np.array([test_image_list, test_label_list])
    test_temp = test_temp.transpose()
    np.random.shuffle(test_temp)

    train_image_list = list(train_temp[:, 0])
    train_label_list = list(train_temp[:, 1])
    train_label_list = [int(i) for i in train_label_list]
    test_image_list = list(test_temp[:, 0])
    test_label_list = list(test_temp[:, 1])
    test_label_list = [int(i) for i in test_label_list]

    return train_image_list, train_label_list, test_image_list, test_label_list


def get_batch(image,label,image_W,image_H,batch_size,capacity=32):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    image = tf.image.per_image_standardization(image)

    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=16,capacity = capacity)

    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch


def inference(images, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE) as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


N_CLASSES = 10
IMG_W = 523
IMG_H = 396
BATCH_SIZE = 1
CAPACITY = 64
MAX_STEP = 2000
learning_rate = 0.005


def run_training():
    data_dir = "../temp/"
    logs_train_dir = 'log/'
    train, train_label, test, test_label = get_files(data_dir, 0.8)
    train_batch, train_label_batch = get_batch(train, train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    # test, test_label = get_files(test_dir)
    test_batch, test_label_batch = get_batch(test, test_label,
                                               IMG_W,
                                               IMG_H,
                                               BATCH_SIZE,
                                               CAPACITY)

    train_logits = inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    test_logits = inference(test_batch, BATCH_SIZE, N_CLASSES)
    test_acc = evaluation(test_logits, test_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)

    try:
        start = time.time()
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss, tra_acc, tes_acc = sess.run([train_op, train_loss, train_acc, test_acc])
            if step % 50 == 0 or step == MAX_STEP-1:
                # print the result every 50 steps
                print('Step %d,train loss = %.6f,train accuracy = %.2f%%,test accuracy = %.2f%%' % (step, tra_loss, tra_acc*100.0, tes_acc*100.0))
                print("elapsed time = %.2f s" % (time.time() - start))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)

            if step % 200 ==0 or (step +1) == MAX_STEP:
                # save the network every 200 steps
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# def get_one_image(img_dir):
#     image = Image.open(img_dir)
#     # Image.open()
#     # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
#     plt.imshow(image)
#     image = image.resize([208, 208])
#     image_arr = np.array(image)
#     return image_arr
#
# def test(test_file):
#     log_dir = 'log/'
#     image_arr = get_one_image(test_file)
#
#     with tf.Graph().as_default():
#         image = tf.cast(image_arr, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         image = tf.reshape(image, [1,208, 208, 3])
#         print(image.shape)
#         p = inference(image,1,5)
#         logits = tf.nn.softmax(p)
#         x = tf.placeholder(tf.float32,shape = [208,208,3])
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             ckpt = tf.train.get_checkpoint_state(log_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 #调用saver.restore()函数，加载训练好的网络模型
#
#                 print('Loading success')
#             else:
#                 print('No checkpoint')
#             prediction = sess.run(logits, feed_dict={x: image_arr})
#             max_index = np.argmax(prediction)
#             print('预测的标签为：')
#             print(max_index)
#             print('预测的结果为：')
#             print(prediction)


run_training()