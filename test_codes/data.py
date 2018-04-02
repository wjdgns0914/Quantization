from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
import tarfile
import gzip
import tensorflow as tf
import math
from six.moves import urllib
# import csv
# import glob
# import re

DATA_DIR = './Datasets/'
URLs = {
    'cifar10': 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    'cifar100': 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz',
    'MNIST_train_image': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'MNIST_train_lable': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'MNIST_test_image': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'MNIST_test_label': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

"""
설명8:
만약에 데이터셋 이름으로 된 폴더가 없을 경우 그 폴더를 일단 만든다,
sys.stdout.flush()명령어는 버퍼를 비워준다,파이썬에서의 출력은 버퍼를 이용한다,
버퍼를 비워준다는건, 버퍼에 있는 데이터가 다 출력(혹은 write) 될 때까지 기다린다는 뜻이다.

\r 이건 carriage return이라는 코드라는데..커서를 항의 맨 앞으로 옮겨주는 역할을 한다고 한다.
즉 매번 stdout.write() 할 때마다 항의 맨 앞에서 다시 쓰는거다, 그래서 한줄에서 퍼센테이지만 변하는 걸로 보인다.

    if apply_func is not None:
        apply_func(filepath)
위의 코드는 apply_func에 파일 압축 푸는 함수를 넣어서, 파일 압축을 풀어준다.
"""

def __maybe_download(data_url, dest_directory, apply_func=None):
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if apply_func is not None:
        apply_func(filepath)

"""
설명6:
cifar데이터를 읽기위한 함수이다.
1)filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle,num_epochs=None)로 읽을 파일들 queue를 만든다.
2)파일들은 읽기편한 정형화 된 데이터가 아니라(.bin확장자), 그냥 비트들의 나열이라고 생각하면 된다.
3)그래서 하나의 데이터가 몇바이트로 이루어져있는지를 계산한다.
4)한번에 계산한 바이트만큼만 읽는 reader를 만든다, 그리고 그 리더로 파일 queue에서 데이터를 읽는다, 파일 queue는 리더가 데이터를
다 읽으면 다음 파일을 넣어주는 queue라고 생각하면 된다.
5)key,value값을 왼쪽에서 받아주는데 key는 사용하지 않는다, 아무래도 자동으로 생성되는 index정도? 아닐까?
6)읽어 온 값을 저장한 value를 tf.decode_raw()함수를 이용해서 두번째인자인 tf.uint8만큼의 양만큼을 한 열로 하는 벡터로 만들어준다.
예를 들자면 이렇다. 만약 2바이트씩 읽는다면 value는 지금 16개의 0혹은 1로 이루어진 string인데, 그걸 2bit씩 즉 tf.decode_raw(value,2bit)
라고 한다면 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1->[[0,0],[1,1],[0,0],[1,1],[0,0],[1,1],[0,0],[1,1]]로 바뀐다(2*8)행렬이라고 보면 된다.
여기서 1 dimension 행렬을 string, 2 dimension 행렬을 vector라고 칭하는 것에 유의하자.
설명:A Tensor of type out_type. A Tensor with one more dimension than the input bytes. The added dimension will have size 
    equal to the length of the elements of bytes divided by the number of bytes to represent out_type.
7)그걸 앞부분 조금 잘라서 label이라고 주고..그 다음 부분을 image라고 준다.
8) 마지막에 traspose를 해서 일반적인 이미지 데이터형식으로 변경한다, 아무래도 데이터가 원래 들어가있는 순서가..조금 특이했나보다.
"""

def __read_cifar(filenames, shuffle=True, cifar100=False):
  """Reads and parses examples from CIFAR data files.
  """
  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle,num_epochs=None)

  label_bytes = 1  # 2 for CIFAR-100
  if cifar100:
      label_bytes = 2
  height = 32
  width = 32
  depth = 3
  image_bytes = height * width * depth
  # Every record consists of a label followed by the image, with
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),  #def slice(input_, begin, size, name=None):
                           [depth, height, width])

  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.transpose(depth_major, [1, 2, 0])

  return tf.cast(image, tf.float32), label

def __read_MNIST(training=True):
    """Reads and parses examples from MNIST data files."""
    mnist = input_data.read_data_sets("Datasets/MNIST", one_hot=False)
    if training:
        images=tf.cast(tf.convert_to_tensor(mnist.train.images.reshape([55000,28,28,1])),tf.float32)
        labels=tf.cast(tf.convert_to_tensor(mnist.train.labels),dtype=tf.int32)
    else:
        images = tf.cast(tf.convert_to_tensor(mnist.test.images.reshape([10000,28,28,1])),dtype=tf.float32)
        labels = tf.cast(tf.convert_to_tensor(mnist.test.labels),dtype=tf.int32)
    return images,labels

    # mnist = input_data.read_data_sets("Datasets/MNIST", one_hot=False)
    # if training:
    #     images=tf.cast(tf.convert_to_tensor(mnist.train.images.reshape([55000,28,28,1])),tf.float32)
    #     labels=tf.cast(tf.convert_to_tensor(mnist.train.labels),dtype=tf.int32)
    # else:
    #     images = tf.cast(tf.convert_to_tensor(mnist.test.images.reshape([10000,28,28,1])),dtype=tf.float32)
    #     labels = tf.cast(tf.convert_to_tensor(mnist.test.labels),dtype=tf.int32)
    # return images,labels

"""
설명5:
클래스 안에 매서드가 한개있다! 클래스응용에 참 좋은 예제인 것 같다.
별거 없다. 데이터셋 받아서 그걸 일정 배치사이즈로 shuffle해주는거다.
min_queue_examples=1000, num_threads=8 는 미리 정해준다, 그리고 웬만하면 이 초깃값을 이용해주는 듯 하다.
만약 배치사이즈가 100이라면, 1300개를 queue에 쌓아놓고, 거기서 100개를 뽑아준다. 
아닌가? num_threads=8이니까 한번에 800개를 뽑으려나..? #미해결
어쩄든 min_queue_examples 가 크면 클수록 큰사이즈의 queue에서 데이터를 뽑게 되므로 잘섞어서 뽑는 셈이 된다.
"""

class DataProvider:
    def __init__(self, data, size=None, training=True,MNIST=False):
        self.size = size or [None]*4
        self.data = data
        self.training = training
        self.enqueue_many=MNIST

    def next_batch(self, batch_size, min_queue_examples=3000, num_threads=8):
        """Construct a queued batch of images and labels.

        Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.

        Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.

        image, label = self.data
        if self.training:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples, enqueue_many=self.enqueue_many)
        else:
            images, label_batch = tf.train.batch(
                [preprocess_evaluation(image, height=self.size[1], width=self.size[2]), label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=min_queue_examples + 3 * batch_size, enqueue_many=self.enqueue_many)

        return images, tf.reshape(label_batch, [batch_size])
"""
설명7:
아래에서 이미지의 전처리를 진행한다. 바로 위 함수를 보면 전처리에 대한 조건없이 모두 전처리를 진행해주는데,
또 아래 코드를 보면 전처리가 무효한 것도 아니다. 즉 데이터셋에 대한 전처리가 진행 중이었던 것이다.
논문에서는 없다고 그랬는데? 이정도는 전처리도 아닌 것인가?
포인트: 인풋으로 들어오는 img는 사진 한장!이다. 여기 나오는 함수들은 데이터셋을 통쨰로 주고받는 것이 아니다.
queue를 구성하고 그 입출력선을 주고받는다.

"""
def preprocess_evaluation(img, height=None, width=None, normalize=None):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]
    preproc_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
    if normalize:
         # Subtract off the mean and divide by the variance of the pixels.
        preproc_image = tf.image.per_image_whitening(preproc_image)
    return preproc_image

def preprocess_training(img, height=None, width=None, normalize=None):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(img, [height, width, img_size[2]])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                             max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                           lower=0.2, upper=1.8)

    # float_image = tf.image.per_image_standardization(distorted_image)

    if normalize:
        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_whitening(distorted_image)
    return distorted_image

"""
설명4:
아래 함수를 통해 데이터를 공급 받는다. 함수는 이름(데이터셋)과 training여부를 인풋으로 받고
데이터셋 이름에 따라 두가지 상황으로 나뉜다.
왜 굳이굳이 나누어놓았냐면 다른 부분은 다 똑같은데 DataProvider()함수 실행하는 부분이 다르기 때문이다.
일단 표면상으로는 그러한데 cifar100부분이 굳이 달라야 할 이유가 있나..? 나중에 cifar100 할 때 알아보자.
MNIST dataset을 추가하려면 여기다가 하면 될거 같다.
포인트: 여기서부터 눈여겨 보아야할건 이 코드에는 클래스(혹은 함수)를 return하는 경우가 상당히 많다는거다, 내가 아직 못배운 사고회로이다.
여기서 이게 왜 유용하냐면 단순히 데이터셋을 반환하면 그걸 또 shuffle해줘야하잖아?
그니까 셔플기능까지 넣은 데이터셋을 반환하고싶은거다. 마치 텐서플로우에 내장되어있는 dataprovider처럼!
"""
def get_data_provider(name,training=True):
    if name == 'cifar10':
        path = os.path.join(DATA_DIR,'cifar10')
        url = URLs['cifar10']
        def post_f(f): return tarfile.open(f, 'r:gz').extractall(path)   #Open for reading with gzip compression.
        __maybe_download(url, path,post_f)
        data_dir = os.path.join(path, 'cifar-10-batches-bin/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'data_batch_%d.bin' % i)
                                    for i in range(1, 6)]), [50000, 32,32,3], True,True,num_threads=8)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test_batch.bin')]),
                                [10000, 32,32, 3], False,True,num_threads=8)
    elif name == 'cifar100':
        path = os.path.join(DATA_DIR,'cifar100')
        url = URLs['cifar100']
        def post_f(f): return tarfile.open(f, 'r:gz').extractall(path)
        __maybe_download(url, path,post_f)
        data_dir = os.path.join(path, 'cifar-100-batches-bin/')
        if training:
            return DataProvider([os.path.join(data_dir, 'train.bin')],50000, True, __read_cifar)
        else:
            return DataProvider([os.path.join(data_dir, 'test.bin')],10000, False, __read_cifar)


    elif name == 'MNIST':
        if training:
            return DataProvider(__read_MNIST(training=True), size=[55000, 28, 28, 1], training=True, MNIST=True)
        else:
            return DataProvider(__read_MNIST(training=False), size=[10000, 28, 28, 1], training=False,
                                MNIST=True)

        # if training:
        #     return DataProvider(__read_MNIST(training=True),size=[55000, 28,28, 1], training=True,argum=False)
        # else:
        #     return DataProvider(__read_MNIST(training=False),size=[10000, 28,28, 1],training=False,argum=False)

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img