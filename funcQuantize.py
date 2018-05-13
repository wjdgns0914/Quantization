import tensorflow as tf
import numpy as np
import funcCustom
from params import *

FLAGS = tf.app.flags.FLAGS
W_level=FLAGS.W_target_level
Wq_level=FLAGS.Wq_target_level

# Func: Calculate normalization factor
# Because the half is in positive range, the other half is in negative field.
# so max(|x|)=2.0 ** (bits - 1), 'x' is the input data, 'bits' is the bits we need to express the input data.
def scalebybits(bits):
    return 2.0 ** (bits - 1)

# Func: Express data in the power of 2
# 1e-12 is for avoiding log(0)
def express_2power(x):
    return 2.0 ** tf.round(tf.log(x+1e-12) / tf.log(2.0))

# Func: Quantize [-1,1] to target_level, which expressed in power of 2
# Q:Scale factor를 꼭 아래와 같이 해야하는가? 즉 보통의 양자화는 양자화 오차를 최소화하고자 하는데
# 딥러닝에서는 레벨간의 구분이 중요할 뿐 절대값이 중요하지 않다면 양자화 오차를 고려하지 않아도 되는가?
# Q: 비선형양자화는 어떻게 추가를 할 것인가?
def quantize(x, target_level=2):
    assert np.floor(target_level) == target_level, "Target_level should be a integer"
    bits = np.ceil(np.log(target_level) / np.log(2.0))
    SCALE = scalebybits(bits)
    if bits > 15:
        print("The target_level is to high, We don't quantize this.")
        y = x
    elif bits == 1:  # BNN
        print("The target_level=1 and we use BNN")
        y = tf.sign(x+0.000001)
    else:
        if target_level % 2 == 1:
            print('target_level is a odd number')
        elif target_level % 2 == 0:
            print('target_level is a even number')
        else:
            raise SystemError("target_level is not valid number")
        y = x*0.5+0.5
        y = tf.round(y*(target_level-1))/(2*SCALE)
        y = 2*y-(target_level-1)/(2*SCALE)
    return tf.stop_gradient(y - x) + x

def clip(x, target_level=None):
    # 이게 왜 필요하냐?
    # 레벨 수가 몇이든 간에 결국 해당하는 bits의 2의 승수배로 weight 값을 주게된다.
    # 그니까 그 최댓값을 넘어가는 값은 다 최댓값으로 맵핑시킬 필요가 있다.
    if target_level==None:
        MAX=1.
        MIN=-1.
    else:
        assert np.floor(target_level) == target_level, "Target_level should be a integer"
        bits = np.ceil(np.log(target_level) / np.log(2.0))
        SCALE = scalebybits(bits)
        if bits > 15 or bits == 1:
            delta = 0.
        elif target_level % 2 == 1:
            delta = (SCALE - np.floor(target_level/2))/SCALE
        else:
            delta = (SCALE*2 - target_level + 1)/(SCALE*2)
        MAX = +1 - delta
        MIN = -1 + delta
    y = tf.clip_by_value(x, MIN, MAX, name='saturate')
    return tf.stop_gradient(y - x) + x

def quantize_G(x, target_level=W_level):
    bitsG = np.ceil(np.log(target_level) / np.log(2.0))
    SCALE = scalebybits(bitsG)
    with tf.name_scope('Quantize_G'):
        if bitsG > 15:
            return x
        else:
            if x.name.lower().find('batchnorm') > -1:
                return x  # batch norm parameters, not quantize now
            xmax = tf.reduce_max(tf.abs(x))
            y = x / express_2power(xmax)
            norm = quantize(LR * y, 4095)
            norm_sign = tf.sign(norm)
            norm_abs = tf.abs(norm)
            norm_int = tf.floor(norm_abs)
            norm_float = norm_abs - norm_int
            rand_float = tf.random_uniform(x.get_shape(), 0, 1)
            norm = norm_sign * (norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1))
            return norm / SCALE

def quantize_W(x, target_level=Wq_level):
    bitsW = np.ceil(np.log(target_level) / np.log(2.0))
    with tf.name_scope('Quantize_W'):
        if bitsW > 15:
            return x
        else:
            xmax = tf.reduce_max(tf.abs(x))
            y = x / express_2power(xmax)
            y = quantize(clip(y), target_level)
            return x + tf.stop_gradient(y - x)  # skip derivation of Quantize and Clip
#a./(1+exp(-b*(info_volt-c)))-d;
def write_memory(x,cell_info,write_info,choice=0,target_level=W_level,Drift=False):  #cell_info=[a,b,c,d,std_val]
    #기본정보, 신경 쓸 필요 없음
    tf.add_to_collection("Original_Weight", x)    # stage1
    filter_shape = x.get_shape().as_list()
    bits = np.ceil(np.log(target_level) / np.log(2.0))
    if bits<12:
        SCALE = scalebybits(bits)
        level_index = tf.to_int32((x * 2 * SCALE + target_level - 1) / 2)
        a,b,c,d,std_val = cell_info
        volt,r,r_ref = write_info
        # 오늘: 이 부분 variable빼고 placeholder랑 numpy로만 할 수 없을까
        pre_tr = \
            tf.Variable(initial_value=tf.ones(shape=filter_shape) * r[0], name='pre_target_resistance', trainable=False)
        pre_tr_place = tf.placeholder(dtype=tf.float32, shape=filter_shape)
        pre_tr_update_op = pre_tr.assign(pre_tr_place)
        pre_wr = \
            tf.Variable(initial_value=tf.ones(shape=filter_shape) * r[0], name='pre_written_resistance', trainable=False)
        pre_wr_place = tf.placeholder(dtype=tf.float32, shape=filter_shape)
        pre_wr_update_op = pre_wr.assign(pre_wr_place)
        tf.add_to_collection('pre_Wbin', pre_tr_place)
        tf.add_to_collection('pre_Wbin_update_op', pre_tr_update_op)
        tf.add_to_collection('pre_Wfluc', pre_wr_place)
        tf.add_to_collection('pre_Wfluc_update_op', pre_wr_update_op)
        # 1번 방법: 무조건 쓴다, 이게 conventional method, 32비트에서 delta를 반영하기 위한 비트만 변경을 한다.
        # 2번 방법: read를 하고 write여부를 결정한다.
        # pre_wr_index=customfunc.make_index(pre_wr,r_ref)
        # tr_index = customfunc.make_index(tr, r_ref)
        # keep_bool=tf.to_float(tf.equal(pre_wr_index,tr_index))
        # 3번 방법: pre-iteration의 target resistance를 기준으로 하여 이번 cycle의 target resistance가 변했는지로 write 여부를 결정
        # pre-iteration의 target resistance는 어디에 저장 할 것인가? 사실 이게 제일 비현실적인 듯.
        # if mode==True and FLAGS.Celltocellvariation==True:
        target_volt=tf.gather(volt, level_index, name='Target_volt')
        tr = tf.floor(a/(1+tf.exp(-b*(target_volt-c)))-d,name='1/Target_resistance')   #target resistance
        std_val_1d = tf.reshape(std_val, shape=[std_val.get_shape().num_elements()])
        level_index_1d = tf.reshape(level_index, shape=[level_index.get_shape().num_elements()]) \
                         + tf.range(0, level_index.get_shape().num_elements(), 1) * W_level
        tr_var = tf.reshape(tf.gather(std_val_1d, level_index_1d), shape=filter_shape)
        # else:
        #     tr = tf.gather(r,level_index,name='1/Target_resistance')
        #     tr_var = tf.gather(std_val,level_index)
        keep_bool = tf.to_float(tf.equal(pre_tr, tr))
        update_bool = 1 - keep_bool
        # if FLAGS.Inter_variation_options==True:
        fluc_value = tf.reshape(tf.distributions.Normal(loc=0., scale=tr_var).sample(1), shape=filter_shape)
        wr=keep_bool*pre_wr+update_bool*(tr+fluc_value)
        # else:
        #     wr=tr
        if (Drift==True)and(W_level==2):
            drift_value = np.array(np.ones(shape=filter_shape) * 0.09)  # drift_value가 dvalue이다,즉 power law에서 지수부분
            drift_Meanmean, drift_Meanstd = 0.09, 0.001
            drift_Stdmean, drift_Stdstd = 0, 0.0003
            new_value = funcCustom.get_distrib([drift_Meanmean, drift_Meanstd, drift_Stdmean, drift_Stdstd], filter_shape)
            new_drift = tf.cast((pre_wr <= r_ref[0]), dtype=tf.float32) * tf.cast((wr > 0), dtype=tf.float32)
            # 전사이클에서는 -1이었고 이번 사이클에 1로 업데이트 된 element 부분을 1로, 즉 한번 set->reset이 될 때마다 d value가 바뀐다.
            # note1:사실 매 iteration drift 할 때마다 d value조차도 약간 바뀌지만 그건 일단 고려하지 않았다.
            # note2:여기서의 drift는 순전히 2levels일 때만 고려,그 이상의 MLC에서는 drift를 사용하지않기를 추천(unrealistic).
            drift_value = tf.identity(new_value * new_drift + drift_value * (1 - new_drift),name='/5/drift value')
            step = tf.Variable(tf.zeros(shape=filter_shape, dtype=tf.float32), trainable=False)
            step = tf.assign(step, keep_bool * (step + 1), name="Drift_step")
            # step은 각 웨이트가 드리프트를 지금 몇스텝째 하고있는지를 나타낸다.
            drift_factor = (step + 1.) / (tf.cast(tf.equal(step, 0.), dtype=tf.float32) + step)
            # step=0인 부분은 결국 분모=1+0=1, step!=0인 부분은 분모=0+step=step
            drift_scale = (tf.log(drift_factor) / tf.log(10.)) * drift_value
            # log가 들어가는 이유는 이 코드 작성 당시 R->W의 맵핑을 log함수라고 생각했었기 때문이다. 지금은 로그맵핑을 쓰지는않을 것 같아서
            # MAC에서 맵핑을 어떻게 하느냐에 따라 수정이 필요할거 같다.
            with tf.control_dependencies([drift_scale]):
                wr = wr+tf.cast(pre_wr>r_ref[0], tf.float32)*keep_bool*drift_scale
            tf.add_to_collection("Drift_value", drift_value)
            tf.add_to_collection("Drift_step", step)
        else:
            wr = wr
        # read phase
        wr = tf.identity(wr, name='2/Written_resistance')
        rl = tf.to_float(funcCustom.make_index(wr,r_ref))
        rl = (2*rl-(target_level-1))/(2*SCALE)
        # rl =x
        tf.add_to_collection('Binarized_Weight', tr)  # stage2/target_resistance
        tf.add_to_collection('Fluctuated_Weight', wr)  # stage3
    else:
        rl=x
    rl = tf.identity(rl,name='3/Read_weight')
    tf.add_to_collection('Read_Weight', rl) # stage4
    return tf.stop_gradient(rl-x)+x
#오늘: scale factor 영향력 알아보자 ,-0.5 0.5   -3, 3이런거  심지어 3,6 이런거

