import tensorflow as tf
import build_Geo_data
def decode_from_tfrecords(tfRecord_path,image_size):
    '''
    生成文件序列读取图片数据
    :param tfRecord_path:
    :param image_size:
    :return:
    '''
    filename_queue = tf.train.string_input_producer(tfRecord_path, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "image/image_id": tf.FixedLenFeature([], tf.int64),
                                           "image/data": tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.image.decode_jpeg(features['image/data'])
    #image = tf.image.convert_image_dtype(image, dtype = tf.uint8)
    image = tf.image.resize_images(image, [image_size, image_size])
    label = tf.cast(features['image/image_id'], tf.int64)
    image = image/255
    return image, label


def get_tfrecord(num, image_size):
    '''
    根据batch获取数据
    :param num:
    :return:
    '''
    #获取文件名列表
    data_files = tf.gfile.Glob(build_Geo_data.FLAGS.input_file_pattern)
    img, label = decode_from_tfrecords(data_files,image_size)
    #label需要进行one-hot编码处理
    label  = tf.one_hot(label, 74)
    #显示表示维度
    img = tf.reshape(img, tf.stack([image_size, image_size, 3]))
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=100,
                                                    min_after_dequeue=70)

    return img_batch, label_batch