from data import *
import customfunc
tf.set_random_seed(333)  # reproducibility
MOVING_AVERAGE_DECAY = 0.999
WEIGHT_DECAY_FACTOR = 0.0001
daystr,timestr=customfunc.beautifultime()
FLAGS = tf.app.flags.FLAGS
# Basic model parameters which will be often modified.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 200,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('dataset', 'MNIST',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('model','MLP_00_Basic',    #'Probe_MLP',MLP_00_Basicvanila  MLP_00_Basic_512*3
                           """Name of loaded model.""")
# Level2 Parameters which will be sometimes modified.
tf.app.flags.DEFINE_integer('W_target_level', 2**32-1,
                            """Target level.""")
tf.app.flags.DEFINE_integer('Wq_target_level', 2,
                            """Target level.""")
tf.app.flags.DEFINE_float('target_std', 0.,
                            """Target std.""")
tf.app.flags.DEFINE_bool('summary', True,                   #Log only include accuracy data
                           """Record summary.""")
tf.app.flags.DEFINE_bool('summary2', True,                 #Log also include histogram, weight's scalar graph..etc
                           """Record summary2.""")
tf.app.flags.DEFINE_bool('Variation', True,
                           """Variation or Not.""")
tf.app.flags.DEFINE_bool('Drift1', False,
                           """Drift or Not.""")
tf.app.flags.DEFINE_bool('Drift2', False,
                           """Drift or Not.""")
tf.app.flags.DEFINE_bool('Weight_decay', False,
                           """Weightdecay or Not.""")
tf.app.flags.DEFINE_bool('Fine_tuning', False,            #FC layer만 작동시킴
                           """Fine_tuning or Not.""")
tf.app.flags.DEFINE_bool('Load_checkpoint', False,        #체크포인트를 불러 올 것인가?
                           """Load_checkpoint or Not.""")
load_checkpoint_path='./results/today/05cifar10_BNN_big2(Drift1_Var_125epoch_81%)'
tf.app.flags.DEFINE_string('load', '',
                           """Name of loaded dir.""")
# Level3 Parameters which will be seldom modified.
tf.app.flags.DEFINE_string('save', timestr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_bool('bit_deterministic', False,       # -1,1,1.4로 양자화 함(2비트)
                           """2bit_deterministic or Not.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './results/'+daystr+'/'+FLAGS.save+str(FLAGS.W_target_level)+'_'+FLAGS.model
                           +'_('+FLAGS.dataset+')_'+str(FLAGS.W_target_level)+'levels',
                           """Constant""")
tf.app.flags.DEFINE_string('log_dir', FLAGS.checkpoint_dir + '/log/',
                           """Constant""")

# LR = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
# LR_schedule = [0, 8, 200, 1,250,1./8,300,0]