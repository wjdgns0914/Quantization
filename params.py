from funcData import *
import funcCustom
tf.set_random_seed(333)  # reproducibility
MOVING_AVERAGE_DECAY = 0.999
WEIGHT_DECAY_FACTOR = 0.0001
daystr,timestr=funcCustom.beautifultime()
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
# 2-1 Quantization options
tf.app.flags.DEFINE_integer('W_target_level', 14,
                            """Target level.""")
tf.app.flags.DEFINE_integer('Wq_target_level',  2,
                            """Target level.""")

# 2-2 Logging data
tf.app.flags.DEFINE_bool('summary', True,                   #Log only include accuracy data
                           """Record summary.""")
tf.app.flags.DEFINE_bool('summary2', True,                 #Log also include histogram, weight's scalar graph..etc
                           """Record summary2.""")
tf.app.flags.DEFINE_integer('choice',0,"""select which type to choose""")
tf.app.flags.DEFINE_integer('col_choice',FLAGS.W_target_level,"""select which type to choose""")
# 2-3 Properties
tf.app.flags.DEFINE_string('Inter_variation_options', "False,False,1,False,1"
                           , """Include Cell-to-cell(Inter-cell) variation or not.""")
tf.app.flags.DEFINE_string('Intra_variation_options', "False,False,1,False,1"
                           , """In-cell(Intra-cell) variation or not""")
# 1. Use variation or not,
# 2. if use variation, use random number or true param   3. if use random number, write a targeted variation
# 4. if use variation, adjust variation or not           5. if adjust variation, write a coefficient using in adjusting

tf.app.flags.DEFINE_bool('Load_ref', False,
                           """Load reference levels or Not.""")
tf.app.flags.DEFINE_bool('Drift1', False,
                           """Drift or Not.""")
tf.app.flags.DEFINE_bool('Drift2', False,
                           """Drift or Not.""")


# Level3 Parameters which will be seldom modified.
# 3-1 Path options
tf.app.flags.DEFINE_string('daystr', daystr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './results/'+daystr+'/'+timestr+'_'+FLAGS.model
                           +'_('+FLAGS.dataset+')_'+str(FLAGS.W_target_level)+'levels'+'_'+str(FLAGS.Wq_target_level)+'levels',
                           """Constant""")
tf.app.flags.DEFINE_string('log_dir', FLAGS.checkpoint_dir + '/log/',
                           """Constant""")
tf.app.flags.DEFINE_string('load', '',
                           """Name of loaded dir.""")

# 3-2 Training methods
tf.app.flags.DEFINE_bool('bit_deterministic', False,       # -1,1,1.4로 양자화 함(2비트)
                           """2bit_deterministic or Not.""")
tf.app.flags.DEFINE_bool('Weight_decay', False,
                           """Weightdecay or Not.""")
tf.app.flags.DEFINE_bool('Fine_tuning', False,            #FC layer만 작동시킴
                           """Fine_tuning or Not.""")
tf.app.flags.DEFINE_bool('Load_checkpoint', False,        #체크포인트를 불러 올 것인가?
                           """Load_checkpoint or Not.""")

load_checkpoint_path='./results/today/05cifar10_BNN_big2(Drift1_Var_125epoch_81%)'
LR = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
LR_schedule = [0, 8,15,6,30, 4,150,2,200,0]