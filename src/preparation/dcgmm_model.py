''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''

import tensorflow as tf
# import ops as utils
# from GMM_M_Step import GMM_M_Step


class CNN(object):
  def __init__(self, name, config, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
    # tf.reset_default_graph()
    # print(self.reuse)

    with tf.variable_scope(self.name, reuse=self.reuse):
        G_W1 = weight_variable([3, 3, 1, 32], name="G_W1")
        G_b1 = bias_variable([32], name="G_b1")
        
        G_W2 = weight_variable([3, 3, 32, 64], name="G_W2")
        G_b2 = bias_variable([64], name="G_b2")
        
        G_W3 = weight_variable([3, 3, 64, 64], name="G_W3")
        G_b3 = bias_variable([64], name="G_b3")
        
        G_W4 = weight_variable([3, 3, 64, 128], name="G_W4")
        G_b4 = bias_variable([128], name="G_b4")
        
        G_W5 = weight_variable([3, 3, 128, 128], name="G_W5")
        G_b5 = bias_variable([128], name="G_b5")
        
        G_W6 = weight_variable([3, 3, 128, 128], name="G_W6")
        G_b6 = bias_variable([128], name="G_b6")
        
        G_W7 = weight_variable([3, 3, 128, 64], name="G_W7")
        G_b7 = bias_variable([64], name="G_b7")
        
        G_W8 = weight_variable([1, 1, 64, 32], name="G_W8")
        G_b8 = bias_variable([32], name="G_b8")
        
        G_W9 = weight_variable([3, 3, 32, config['ClusterNo']], name="G_W9")
        G_b9 = bias_variable([config['ClusterNo']], name="G_b9")
        
        self.Param = {'G_W1':G_W1, 'G_b1':G_b1, 
                 'G_W2':G_W2, 'G_b2':G_b2,  
                 'G_W3':G_W3, 'G_b3':G_b3, 
                 'G_W4':G_W4, 'G_b4':G_b4, 
                 'G_W5':G_W5, 'G_b5':G_b5, 
                 'G_W6':G_W6, 'G_b6':G_b6, 
                 'G_W7':G_W7, 'G_b7':G_b7, 
                 'G_W8':G_W8, 'G_b8':G_b8, 
                 'G_W9':G_W9, 'G_b9':G_b9 }
      
    if self.reuse is None:
          self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
          self.saver = tf.train.Saver(self.var_list)
          self.reuse = True         

   
  def __call__(self, D):
    # print(self.reuse)
    with tf.variable_scope(self.name, reuse=self.reuse):
        
        D_norm = D 
        
        G_conv1 = conv2d_basic(D_norm, self.Param['G_W1'], self.Param['G_b1'])    # 卷积
        G_relu1 = tf.nn.relu(G_conv1, name="G_relu1")
    
        G_conv2 = conv2d_basic(G_relu1, self.Param['G_W2'], self.Param['G_b2']) # 卷积
        G_relu2 = tf.nn.relu(G_conv2, name="G_relu2")
        
        G_pool1 = max_pool_2x2(G_relu2)
        
        G_conv3 = conv2d_basic(G_pool1, self.Param['G_W3'], self.Param['G_b3'])  # 卷积
        G_relu3 = tf.nn.relu(G_conv3, name="G_relu3")
        
        G_conv4 = conv2d_basic(G_relu3, self.Param['G_W4'], self.Param['G_b4'])   # 卷积
        G_relu4 = tf.nn.relu(G_conv4, name="G_relu4")
        
        G_pool2 = max_pool_2x2(G_relu4)                                          # 池化
        
        G_conv5 = conv2d_basic(G_pool2, self.Param['G_W5'], self.Param['G_b5'])  # 卷积
        G_relu5 = tf.nn.relu(G_conv5, name="G_relu5")
        
        output_shape = G_relu5.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = self.Param['G_W6'].get_shape().as_list()[2]
           
        G_rs6 = tf.image.resize_images(G_relu5, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   # 上采样层
        G_conv6 = conv2d_basic(G_rs6, self.Param['G_W6'], self.Param['G_b6'])  # 卷积
        G_relu6 = tf.nn.relu(G_conv6, name="G_rs6")
        
        output_shape = G_relu6.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = self.Param['G_W7'].get_shape().as_list()[2]
    
        G_rs7 = tf.image.resize_images(G_relu6, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 上采样层
        G_conv7 = conv2d_basic(G_rs7, self.Param['G_W7'], self.Param['G_b7'])   # 卷积
        G_relu7 = tf.nn.relu(G_conv7, name="G_rs7")
        
        G_conv8 = conv2d_basic(G_relu7, self.Param['G_W8'], self.Param['G_b8'])  # 卷积
        G_relu8 = tf.nn.relu(G_conv8, name="G_relu8")
        
        G_conv9 = conv2d_basic(G_relu8, self.Param['G_W9'], self.Param['G_b9'])  # 卷积
        Gama = tf.nn.softmax(G_conv9, name="G_latent_softmax")
        

    return Gama
  

class DCGMM(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train


    self.X_hsd = tf.placeholder(tf.float32, shape=[config['batch_size'], config['im_size'], config['im_size'], 3], name="original_color_image")#网络输入
    self.D, h_s = tf.split(self.X_hsd,[1,2], axis=3)# 将张量切割为[config.batch_size, config.im_size, config.im_size, 1]和[config.batch_size, config.im_size, config.im_size, 2]

    self.E_Step = CNN("E_Step", config, is_train=self.is_train)  # 这是一个自编码器神经网络类
    self.Gama = self.E_Step(self.D)# 输入为self.D：[config.batch_size, config.im_size, config.im_size, 1]，输出Gama为[config.batch_size, config.im_size, config.im_size, config.ClusterNo]并且输出进行了softmax处理
    self.loss, self.Mu, self.Std = GMM_M_Step(self.X_hsd, self.Gama, config['ClusterNo'], name='GMM_Statistics')
    
    if self.is_train:

      self.optim = tf.train.AdamOptimizer(config.lr)
      self.train = self.optim.minimize(self.loss, var_list=self.E_Step.Param)

    # ClsLbl = tf.arg_max(self.Gama, 3)     # 返回某一维度上最大值的索引【1,32,32】
    # ClsLbl = tf.cast(ClsLbl, tf.float32)  # 数据类型转换
    #
    # ColorTable = [[255,0,0],[0,255,0],[0,0,255],[255,255,0], [0,255,255], [255,0,255]]
    # colors = tf.cast(tf.constant(ColorTable), tf.float32)
    # Msk = tf.tile(tf.expand_dims(ClsLbl, axis=3),[1,1,1,3])  #【1,32,32,3】
    # for k in range(0, config.ClusterNo):
    #     ClrTmpl = tf.einsum('anmd,df->anmf', tf.expand_dims(tf.ones_like(ClsLbl), axis=3), tf.reshape(colors[k,...],[1,3]))
    #     Msk = tf.where(tf.equal(Msk,k), ClrTmpl, Msk)

    self.X_rgb = HSD2RGB(self.X_hsd)
    # tf.summary.image("1.Input_image", self.X_rgb*255.0, max_outputs=2)
    # tf.summary.image("2.Gamma_image",  Msk, max_outputs=2)
    # tf.summary.image("3.Density_image", self.D*255.0, max_outputs=2)
    tf.summary.scalar("loss", self.loss)

    self.summary_op = tf.summary.merge_all()

    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(config['logs_dir'], self.sess.graph)

    self.sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(config['logs_dir'])
    if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored.....")
   

  def fit(self, X):
    _, loss, summary_str = self.sess.run([self.train, self.loss, self.summary_op], {self.X_hsd:X})
    return loss, summary_str, self.summary_writer

  def deploy(self, X):
    mu, std, gama, summary_str = self.sess.run([self.Mu, self.Std, self.Gama, self.summary_op], {self.X_hsd:X})
    
    return mu, std, gama
    
  def save(self, dir_path):
    self.E_Step.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.E_Step.restore(self.sess, dir_path+"/model.ckpt")


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)

    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def HSD2RGB(X_HSD):
    X_HSD_0, X_HSD_1, X_HSD_2 = tf.split(X_HSD, [1, 1, 1], axis=3)
    D_R = (X_HSD_1 + 1) * X_HSD_0
    D_G = 0.5 * X_HSD_0 * (2 - X_HSD_1 + tf.sqrt(tf.constant(3.0)) * X_HSD_2)
    D_B = 0.5 * X_HSD_0 * (2 - X_HSD_1 - tf.sqrt(tf.constant(3.0)) * X_HSD_2)

    X_OD = tf.concat([D_R, D_G, D_B], 3)
    X_RGB = 1.0 * tf.exp(-X_OD)
    return X_RGB

def GMM_M_Step(X, Gama, ClusterNo, name='GMM_Statistics', **kwargs):
    D, h, s = tf.split(X, [1, 1, 1], axis=3)

    WXd = tf.multiply(Gama, tf.tile(D, [1, 1, 1, ClusterNo]))  # tf.tile在某一维度进行复制
    WXa = tf.multiply(Gama, tf.tile(h, [1, 1, 1, ClusterNo]))
    WXb = tf.multiply(Gama, tf.tile(s, [1, 1, 1, ClusterNo]))

    S = tf.reduce_sum(tf.reduce_sum(Gama, axis=1), axis=1)
    S = tf.add(S, tf.contrib.keras.backend.epsilon())  # tf.contrib.keras.backend.epsilon()返回一个极小的浮点数
    S = tf.reshape(S, [1, ClusterNo])  # 得到的是聚成每一类的数量

    M_d = tf.div(tf.reduce_sum(tf.reduce_sum(WXd, axis=1), axis=1), S)  # 得到的是每一类的D值的平均数
    M_a = tf.div(tf.reduce_sum(tf.reduce_sum(WXa, axis=1), axis=1), S)  # 得到的是每一类的h值的平均数
    M_b = tf.div(tf.reduce_sum(tf.reduce_sum(WXb, axis=1), axis=1), S)  # 得到的是每一类的s值的平均数

    Mu = tf.split(tf.concat([M_d, M_a, M_b], axis=0), ClusterNo, 1)  # 得到的是某一类的D,h,s值的平均值（一共有四类）【4,3】

    Norm_d = tf.squared_difference(D, tf.reshape(M_d, [1, ClusterNo]))
    Norm_h = tf.squared_difference(h, tf.reshape(M_a, [1, ClusterNo]))
    Norm_s = tf.squared_difference(s, tf.reshape(M_b, [1, ClusterNo]))

    WSd = tf.multiply(Gama, Norm_d)
    WSh = tf.multiply(Gama, Norm_h)
    WSs = tf.multiply(Gama, Norm_s)

    S_d = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSd, axis=1), axis=1), S))
    S_h = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSh, axis=1), axis=1), S))
    S_s = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSs, axis=1), axis=1), S))

    Std = tf.split(tf.concat([S_d, S_h, S_s], axis=0), ClusterNo, 1)  # 得到的是某一类的D,h,s值的std值（一共有四类）【4,3】

    #  计算loss值
    dist = list()
    for k in range(0, ClusterNo):
        dist.append(
            tf.contrib.distributions.MultivariateNormalDiag(tf.reshape(Mu[k], [1, 3]), tf.reshape(Std[k], [1, 3])))

    PI = tf.split(Gama, ClusterNo, axis=3)
    Prob0 = list()
    for k in range(0, ClusterNo):
        Prob0.append(tf.multiply(tf.squeeze(dist[k].prob(X)), tf.squeeze(PI[k])))

    Prob = tf.convert_to_tensor(Prob0, dtype=tf.float32)
    Prob = tf.minimum(tf.add(tf.reduce_sum(Prob, axis=0), tf.contrib.keras.backend.epsilon()),
                      tf.constant(1.0, tf.float32))
    Log_Prob = tf.negative(tf.log(Prob))
    Log_Likelihood = tf.reduce_mean(Log_Prob)

    return Log_Likelihood, Mu, Std