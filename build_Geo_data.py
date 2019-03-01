import random, threading, os, sys
import tensorflow as tf
import numpy as np
from collections import namedtuple, Counter
from datetime import datetime

tf.flags.DEFINE_string("train_image_dir", r"C:\Users\16254\Desktop\机器学习\2012年度岩屑薄片鉴定\鉴定文档\images",
                       "Training image directory.")
tf.flags.DEFINE_string("train_captions_file", r"C:\Users\16254\Desktop\机器学习\2012年度岩屑薄片鉴定\鉴定文档\images\annotation.txt",
                       "Training captions txt file.")
tf.flags.DEFINE_string("output_dir", r"C:\Users\16254\Desktop\机器学习\2012年度岩屑薄片鉴定\鉴定文档\images_tf",
                       "Output data directory.")
tf.flags.DEFINE_string("word_counts_output_file", r"C:\Users\16254\Desktop\机器学习\2012年度岩屑薄片鉴定\鉴定文档\images_tf\word_counts.txt",
                       "Output vocabulary file of word counts.")
tf.flags.DEFINE_integer("train_shards", 8,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 2,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_string("input_file_pattern", r"C:\Users\16254\Desktop\机器学习\2012年度岩屑薄片鉴定\鉴定文档\images_tf\train-?????-of-00008",
                       "File pattern of sharded TFRecord input files.")


FLAGS = tf.flags.FLAGS
ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


def _create_vocab(captions):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
      captions: A list of lists of strings.

    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update([c]) #这里需要统计岩性所以将其强制转换为列表
    # 根据词频进行排序
    word_counts = [x for x in counter.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))
    # 输出词频统计文件以及词典文件
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        #类别从1开始
        f.write("\n".join(["%d %s %d" % (i, w[0], w[1]) for i, w in enumerate(word_counts)]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # 创建岩性词典
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _load_and_process_metadata(caption_file):
    image_metadata = []
    with open(caption_file, 'r') as f:
        for line in f:
            line = line.strip().split(':')
            image_metadata.append(ImageMetadata(line[0], str(line[0])+'.jpeg', line[1]))
    return image_metadata


def _to_sequence_example(imdir, image, vocab):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
      image: An ImageMetadata object.
    Returns:
      A SequenceExample proto.
    """
    filename = os.path.join(imdir, image.filename)
    with tf.gfile.FastGFile(filename, "rb") as f:
        encoded_image = f.read()
    # tf文件中的数据格式
    context = tf.train.Features(feature={
        "image/image_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[vocab.word_to_id(image.captions)])),
        "image/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
    })
    sequence_example = tf.train.SequenceExample(context=context)
    return sequence_example



def _process_image_files(thread_index, ranges, images, vocab, num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.
    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      images: List of ImageMetadata.
      num_shards: Integer number of shards for the output files.
    """
    # 每一个线程处理N个输出文件（output files） N = num_shards / num_threads
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        # 生成每个线程对应的文件名
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % ('train', shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(FLAGS.train_image_dir, image, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 100:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()

    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()



def _process_dataset(images, vocab,num_shards):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
      images: List of ImageMetadata.
      num_shards: Integer number of shards for the output files.
    """

    random.seed(12345)
    random.shuffle(images)
    # 根据线程分割图像数据
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    print(spacing)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # 创建协调器监控线程运行情况
    coord = tf.train.Coordinator()
    # 启动线程
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, images, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # 等待运行结束
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set train." %
          (datetime.now(), len(images)))


def main(unused_argv):
    # 判断进程是否匹配
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")

    # 创建输出目录
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # 加载数据文件
    train_dataset = _load_and_process_metadata(FLAGS.train_captions_file)
    # 根据训练文件创建词汇表
    train_captions = [image.captions for image in train_dataset]
    vocab = _create_vocab(train_captions)

    # 处理数据为tf格式
    _process_dataset(train_dataset, vocab, FLAGS.train_shards)
if __name__ == "__main__":
    tf.app.run()
