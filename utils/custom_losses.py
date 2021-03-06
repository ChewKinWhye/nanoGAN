import tensorflow as tf
from keras import losses


# Average distance from the Center of Mass and binary cross entropy loss
def fence_loss(g_out, beta, power):
    def dispersion_loss(y_true, y_predicted):
        loss_b = tf.reduce_mean(losses.binary_crossentropy(y_true, y_predicted))
        
        center = tf.reduce_mean(g_out, axis=0, keepdims=True)
        distance_xy = tf.pow(tf.abs(tf.subtract(g_out, center)), power)
        distance = tf.reduce_sum(distance_xy, 1)
        avg_distance = tf.reduce_mean(tf.pow(tf.abs(distance), 1/power))
        loss_d = tf.math.reciprocal(avg_distance)
        
        loss = loss_b + beta*loss_d
        return loss
    return dispersion_loss
