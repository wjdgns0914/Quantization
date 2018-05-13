import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
def add_summaries_scalar(scalar_list=[],makeornot=False):
    for var in scalar_list:
        if var != None:
            tf.summary.scalar((var.op.name).split('/')[-1]+'/training_real', var)
def add_summaries_activation(activation_list=[],makeornot=False):
    for activation in activation_list:
        if activation != None:
            tf.summary.histogram(activation.op.name, activation)
def add_summaries_gradient(grad_list=[],makeornot=False):
    for grad, var in grad_list:
        if grad != None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

#Name:
#Quantized weight=Real_Binarized_weight
#Target_resistance=Binarized_weight
#Written_resistance=Fluctuated_weight
#Read_weight=Real_Fluctuated_weight
def add_summaries_weight(weight_list=[],makeornot=False):
    for weight in weight_list:
        if weight != None:
            tf.summary.histogram(weight.op.name, weight)
        num_weights = 5
        name_layer=(weight.op.name).split('/')[0]
        index_history=tf.get_collection(key='index_history/'+name_layer)
        if index_history == []:
            for i in range(num_weights):
                index = [np.random.randint(j) for j in weight.get_shape().as_list()]
                index_history.append(index)
            tf.add_to_collection('index_history/'+name_layer,index_history)
            file = open(FLAGS.checkpoint_dir + "/save_model.py", "a")
            print("'''",name_layer, ' : ', index_history, "'''", file=file)
            file.close()
            index_history = tf.get_collection(key='index_history/' + name_layer)
        for index in index_history[0]:
            index_str = '/' + str(index).replace(', ', '_')[1:-1]+'/'
            tf.summary.scalar(name_layer + index_str + '/'.join((weight.op.name).split('/')[1:]), weight[index])
def add_summaries_drift_info(drift_step_list=[],read_weight_list=[],makeornot=False):
    for step,Wbin in zip(drift_step_list,read_weight_list):
        if step != None:
            name_layer=(step.op.name).split('/')[0]
            num = step.get_shape().num_elements()
            weights_keeping1 = tf.cast(step > 20, dtype=tf.float32)
            weights_keeping2 = tf.cast(step > 120, dtype=tf.float32)
            weights_keeping3 = tf.cast(step > 840, dtype=tf.float32)

            weights_keeping_num1 = tf.reduce_sum(weights_keeping1)
            weights_keeping_num2 = tf.reduce_sum(weights_keeping2)
            weights_keeping_num3 = tf.reduce_sum(weights_keeping3)

            weights_keeping_num_reset1 = tf.reduce_sum(weights_keeping1 * tf.cast(Wbin >= 0, dtype=tf.float32))
            weights_keeping_num_reset2 = tf.reduce_sum(weights_keeping2 * tf.cast(Wbin >= 0, dtype=tf.float32))
            weights_keeping_num_reset3 = tf.reduce_sum(weights_keeping3 * tf.cast(Wbin >= 0, dtype=tf.float32))

            ratio11 = weights_keeping_num1 / num
            ratio12 = weights_keeping_num2 / num
            ratio13 = weights_keeping_num3 / num
            ratio21 = weights_keeping_num_reset1 / tf.reduce_sum(tf.cast(Wbin >= 0, dtype=tf.float32))
            ratio22 = weights_keeping_num_reset2 / tf.reduce_sum(tf.cast(Wbin >= 0, dtype=tf.float32))
            ratio23 = weights_keeping_num_reset3 / tf.reduce_sum(tf.cast(Wbin >= 0, dtype=tf.float32))
            tf.summary.scalar(name_layer + "/Ratio_Keeping/1_125", ratio11)
            tf.summary.scalar(name_layer + "/Ratio_Keeping/1_205", ratio12)
            tf.summary.scalar(name_layer + "/Ratio_Keeping/1_300", ratio13)
            tf.summary.scalar(name_layer + "/Ratio_Drifted/1_125", ratio21)
            tf.summary.scalar(name_layer + "/Ratio_Drifted/1_205", ratio22)
            tf.summary.scalar(name_layer + "/Ratio_Drifted/1_300", ratio23)