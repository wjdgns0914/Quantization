import tensorflow as tf
import params
import sys
import numpy as np
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('bh_siz', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('bh_siz2', 32,
                            """Number of images to process in a batch.""")
print(eval(FLAGS.Inter_variation_options.split(',')[0]))
def main(argv=None):  # pylint: disable=unused-argument
    print("Check1",isinstance(3,bool))
    print("Check2", isinstance(True, bool))
    a="True,3,True,2"
    print("Old a=", a,type(a))
    a=eval('['+a+']')
    print("New a=", a, type(a))
    print(argv)
    a,b,c=[33,44,55]
    print(np.linspace(1.5,4,5))
if __name__ == '__main__':
    # print(callable(main))
    tf.app.run()