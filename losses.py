import tensorflow as tf



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) # we can try dice loss


LAMBDA = 100
def generator_loss(disc_generated_output, gen_output, target):
    #gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_generated_output) - disc_generated_output, 2))


    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    #real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    #generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    
    real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_generated_output) - disc_generated_output, 2))
    generated_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_generated_output) - disc_generated_output, 2))
    
    total_disc_loss = (real_loss + generated_loss)*0.5

    return total_disc_loss


def snr(x,y,y_true):
        tmp_snr = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - y)))
        out = 10.0 * tf.log(tmp_snr) / tf.log(10.0)             # 输出图片的snr

        tmp_snr0 = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - x)))
        out0 = 10.0 * tf.log(tmp_snr0) / tf.log(10.0)           # 输入图片的snr

        del_snr = out - out0
        return del_snr, out
    
 