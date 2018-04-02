"""
20180227
이 코드는 stdmean과 stdstd를 몇정도로 설정을 해야지 레벨끼리 미묘하게 겹치는지! 그걸 알아보기 위한 것이다.
수학적으로 알아보려면 mean을 기준으로 좌우 1std는 몇퍼센트, 2std는 몇퍼센트, 3std는 몇퍼센트 그걸 기준으로 하면 될 것이고
이건 visualization을 위해 작성한 코드이다. 그런데 막상 작성해서 텐서보드를 보니 히스토그램을 정말 안예쁘게 그려주더라.
그래서 매우 실망하였다. 대충 모양만 보고 랩미팅 같은거 할 때는 매트랩으로 그리도록 하자.
"""

import time
import tensorflow as tf
import customfunc

for target_level in range(2,8):

    tf.reset_default_graph()
    time.sleep(1)
    daystr, timestr = customfunc.beautifultime()

    mean_weight=0.
    std_weight=0.7
    std_param=[0., 0., 0.3, 0.002]
    shape=[2000,1000]

    x=tf.distributions.Normal(loc=mean_weight,scale=std_weight).sample(sample_shape=shape)
    x = tf.tanh(x)
    x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
    x1=tf.round(x*(target_level-1))
    x2=x1/(target_level-1)
    x2=2*x2-1
    fluc = x2+customfunc.get_distrib(std_param, shape)

    tf.summary.histogram("Raw data",x)
    tf.summary.histogram("Bin number",x1)
    tf.summary.histogram("Dequantized data",x2)
    tf.summary.histogram("Fluctuated data",fluc)
    summary_op=tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer=tf.summary.FileWriter("./result_small_test/"+daystr+'/'+str(target_level)+
                                             "var({0},{1})".format(std_param[2],std_param[3]),graph=sess.graph)
        summary_writer.add_summary(sess.run(summary_op),global_step=1)
        summary_writer.close()
        print(sess.run([x2,fluc]))
