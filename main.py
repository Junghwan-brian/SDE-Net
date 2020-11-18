import tensorflow as tf
from data import data_loader
from model.SDENet import SDENet
import tensorflow_addons as tfa
from warnings import filterwarnings

filterwarnings("ignore")
epochs = 60
lr = 1e-4
lr2 = 0.01
seed = 0
batch_size = 128
target_scale = 10.939756
# Data
print("==> Preparing data..")

tf.random.set_seed(seed)

X_train, y_train, X_test, y_test = data_loader.load_dataset("MSD")

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

# Model
print("==> Building model..")
model = SDENet(4)

real_label = 0  # training data
fake_label = 1  # out-of-distribution data

criterion = tf.keras.losses.BinaryCrossentropy()


# optimize drift net
optimizer_drift = tfa.optimizers.SGDW(learning_rate=lr, weight_decay=5e-4,)

# optimize fc layer
optimizer_fc = tfa.optimizers.SGDW(learning_rate=lr, weight_decay=5e-4,)

# optimize down sampling layer
optimizer_dsl = tfa.optimizers.SGDW(learning_rate=lr, weight_decay=5e-4)

# optimize diffusion net
optimizer_diffusion = tfa.optimizers.SGDW(learning_rate=lr2, weight_decay=5e-4,)


def nll_loss(y, mean, sigma):
    loss = tf.math.reduce_mean(tf.math.log(sigma ** 2) + (y - mean) ** 2 / (sigma ** 2))
    return loss


mse = tf.keras.losses.MSE

train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()
in_loss = tf.keras.metrics.Mean()
out_loss = tf.keras.metrics.Mean()


@tf.function
def train_net(x, y):
    with tf.GradientTape(persistent=True) as tape:
        mean, sigma = model(x, training_diffusion=False)
        loss = nll_loss(y, mean, sigma)

    drift_gradient = tape.gradient(loss, model.drift.trainable_variables)
    dsl_gradient = tape.gradient(loss, model.downsampling_layers.trainable_variables)
    fc_gradient = tape.gradient(loss, model.fc_layers.trainable_variables)

    drift_gradient = [(tf.clip_by_norm(grad, 100)) for grad in drift_gradient]
    dsl_gradient = [(tf.clip_by_norm(grad, 100)) for grad in dsl_gradient]
    fc_gradient = [(tf.clip_by_norm(grad, 100)) for grad in fc_gradient]

    optimizer_drift.apply_gradients(
        zip(drift_gradient, model.drift.trainable_variables)
    )
    optimizer_dsl.apply_gradients(
        zip(dsl_gradient, model.downsampling_layers.trainable_variables)
    )
    optimizer_fc.apply_gradients(zip(fc_gradient, model.fc_layers.trainable_variables))
    train_loss(loss)


@tf.function
def train_diffusion(real_x):
    with tf.GradientTape(watch_accessed_variables=False) as real_tape_diffusion:
        # only access to diffusion layer's parameters
        real_tape_diffusion.watch(model.diffusion.trainable_variables)
        real_y = tf.fill((real_x.shape[0], 1), real_label)
        real_pred = model(real_x, training_diffusion=True)
        real_loss = mse(real_y, real_pred)

    diffusion_gradient = real_tape_diffusion.gradient(
        real_loss, model.diffusion.trainable_variables
    )

    diffusion_gradient1 = [(tf.clip_by_norm(grad, 100)) for grad in diffusion_gradient]

    with tf.GradientTape(watch_accessed_variables=False) as fake_tape_diffusion:
        fake_tape_diffusion.watch(model.diffusion.trainable_variables)
        # fake std is 2 in official code, but in paper it is 4
        fake_x = (
            tf.cast(
                tf.random.normal((real_x.shape[0], 90), mean=0, stddev=2), "float64"
            )
            + real_x
        )
        fake_y = tf.fill((real_x.shape[0], 1), fake_label)
        fake_pred = model(fake_x, training_diffusion=True)
        fake_loss = mse(fake_y, fake_pred)

    diffusion_gradient = fake_tape_diffusion.gradient(
        fake_loss, model.diffusion.trainable_variables
    )

    diffusion_gradient2 = [(tf.clip_by_norm(grad, 100)) for grad in diffusion_gradient]

    optimizer_diffusion.apply_gradients(
        zip(diffusion_gradient1, model.diffusion.trainable_variables)
    )
    optimizer_diffusion.apply_gradients(
        zip(diffusion_gradient2, model.diffusion.trainable_variables)
    )
    in_loss(real_loss)
    out_loss(fake_loss)


@tf.function
def test_net(x, y):
    current_mean = 0
    for i in range(10):
        mean, sigma = model(x, training_diffusion=False)
        current_mean += mean
    current_mean = current_mean / 10
    loss = mse(y, current_mean) * target_scale

    test_loss(loss)


# Training
def train():

    for data, label in train_ds:
        train_net(data, label)
        train_diffusion(data)

    print(
        f"Loss: {train_loss.result():.4f}, Loss_in: {in_loss.result():.4f}, Loss_out: {out_loss.result():.4f}"
    )
    train_loss.reset_states()
    in_loss.reset_states()
    out_loss.reset_states()


# Testing
def test():
    for data, label in test_ds:
        test_net(data, label)
    print(f"Test Loss : {test_loss.result():.4f}")
    test_loss.reset_states()


for epoch in range(epochs):
    print(f"\nEpoch: {epoch + 1}")

    if epoch == 0:
        # In the thesis, initial sigma_max is 0.01 but in the author's code it is 0.1
        model.sigma_max = 0.1
    if epoch == 30:
        model.sigma_max = 0.5
    train()
    test()

path = "sde_net.h5"
tf.saved_model.save(model, path)
