import tensorflow as tf
import GeoC_lenet_forward, utils
import os
import sys

BATCH_SIZE = 10
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = os.path.join(sys.path[0],'tmp')
MODEL_NAME = 'GeoC_lenet_model'
train_num_examples = 1800

def backward():

    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        GeoC_lenet_forward.IMAGE_SIZE,
        GeoC_lenet_forward.IMAGE_SIZE,
        GeoC_lenet_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, GeoC_lenet_forward.OUTPUT_NODE])
    y = GeoC_lenet_forward.forward(x,True,REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    img_batch, label_batch = utils.get_tfrecord(BATCH_SIZE, GeoC_lenet_forward.IMAGE_SIZE)

    with tf.Session() as sess:


        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])  # 6
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)


def main():
    backward()

if __name__ == '__main__':

    main()






