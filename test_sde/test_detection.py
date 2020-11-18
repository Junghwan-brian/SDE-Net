from test_sde.calculate_log import *
import tensorflow as tf
from data.data_loader import load_dataset

# Training settings
eval_iter = 10
seed = 0
gpu = 0
outf = "test_sde"
droprate = 0.1
batch_size = 512
target_scale = 10.939756
Iter_test = 100

tf.random.set_seed(seed)


X_train, y_train, X_test, y_test = load_dataset("MSD")

in_test_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

X_out = load_dataset("boston")


path = "sde_net.h5"
model = tf.keras.models.load_model(path, compile=False)


def mse(y, mean):
    loss = tf.math.reduce_mean((y - mean) ** 2)
    return loss


def generate_target():
    f1 = open("%s/confidence_Base_In.txt" % outf, "w")
    test_loss = 0
    for data, targets in in_test_ds:
        current_mean = 0
        for j in range(eval_iter):
            mean, sigma = model(data)
            current_mean = mean + current_mean
            if j == 0:
                Sigma = tf.expand_dims(sigma, 1)
                Mean = tf.expand_dims(mean, 1)
            else:
                Sigma = tf.concat((Sigma, tf.expand_dims(sigma, 1)), axis=1)
                Mean = tf.concat((Mean, tf.expand_dims(mean, 1)), axis=1)
        current_mean = current_mean / eval_iter
        loss = mse(targets, current_mean)
        test_loss += loss.numpy()
        Var_mean = tf.math.reduce_std(Mean, axis=-1)
        for i in range(data.shape[0]):
            soft_out = Var_mean[i].numpy()
            f1.write("{}\n".format(-soft_out))

    f1.close()

    print("\n Final RMSE: {}".format(np.sqrt(test_loss / Iter_test) * target_scale))


def generate_non_target():
    f2 = open("%s/confidence_Base_Out.txt" % outf, "w")
    current_mean = 0
    for j in range(eval_iter):
        mean, sigma = model(X_out)
        current_mean = mean + current_mean
        if j == 0:
            Sigma = tf.expand_dims(sigma, 1)
            Mean = tf.expand_dims(mean, 1)
        else:
            Sigma = tf.concat((Sigma, tf.expand_dims(sigma, 1)), axis=1)
            Mean = tf.concat((Mean, tf.expand_dims(mean, 1)), axis=1)

    Var_mean = tf.math.reduce_std(Mean, axis=-1)
    for i in range(X_out.shape[0]):
        soft_out = Var_mean[i].numpy()
        f2.write("{}\n".format(-soft_out))
    f2.close()


print("generate log from in-distribution data")
generate_target()
print("generate log  from out-of-distribution data")
generate_non_target()
print("calculate metrics for OOD")
metric(outf, "OOD")
