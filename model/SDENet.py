import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

tf.keras.backend.set_floatx("float64")
tf.random.set_seed(0)


class Drift(Layer):
    def __init__(self):
        super(Drift, self).__init__(name="drift_net")
        self.fc = Dense(50)  # input : 50
        self.relu = ReLU()

    def call(self, t, x):
        out = self.relu(self.fc(x))
        return out


class Diffusion(Layer):
    def __init__(self):
        super(Diffusion, self).__init__(name="diffusion_net")
        self.relu = ReLU()
        self.fc1 = Dense(100)  # input : 50
        self.fc2 = Dense(1)  # input : 100

    def call(self, t, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = tf.nn.sigmoid(out)
        return out  # batch,1


class SDENet(Model):
    def __init__(self, layer_depth):
        super(SDENet, self).__init__(name="SDE_Net")
        self.layer_depth = layer_depth
        self.downsampling_layers = Dense(50)  # batch, 50
        self.drift = Drift()  # batch, 1
        self.diffusion = Diffusion()
        self.fc_layers = Sequential(
            [ReLU(), Dense(2)]
        )  # input : 50, output : mean, variance
        self.deltat = 4.0 / self.layer_depth  # T:4, N:layer_depth
        self.sigma_max = 0.5  # sigma_max : scaling diffusion output

    def call(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma_max * self.diffusion(t, out)
            for i in range(self.layer_depth):
                t = 4 * (float(i)) / self.layer_depth
                out = (
                    out
                    + self.drift(t, out) * self.deltat
                    + diffusion_term
                    * tf.cast(tf.math.sqrt(self.deltat), "float64")
                    * tf.random.normal(tf.shape(out), dtype="float64")
                )  # Euler-Maruyama method

            final_out = self.fc_layers(out)
            mean = final_out[:, 0]
            # sigma should be greater than 0.
            sigma = tf.math.softplus(final_out[:, 1]) + 1e-3
            return mean, sigma

        else:
            t = 0
            final_out = self.diffusion(t, out)
            return final_out
