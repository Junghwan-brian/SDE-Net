# SDE-Net
### Paper: [SDE-Net: Equipping Deep Neural Networks with Uncertainty Estimates](https://arxiv.org/abs/2008.10546)
----------------------------------------------------------------------------------------------
## Requirements
- Tensorflow 2.3 ver.
- tensorflow_addons 0.11.2 ver.
- numpy 1.19.4 ver.
- pandas 1.1.4 ver.
----------------------------------------------------------------------------------------------


## introduction

The traditional methods of estimating uncertainties were mainly Bayesian methods. Bayesian methods should introduce the preor to estimate the distribution of the posteror.
But DNN has too many parameters, so it's hard to calculate. So there are also non-bayesian methods that are most famous for modelling methods. 
This method learns several DNNs to obtain an uncertainty with different degrees of prediction. 
This method has a large computational cost because it requires the learning of several models. 
Other methods have problems that cannot be measured by distinguishing between an entity's data from an entity's internal uncertainty.

Solve these problems using SDE-Net. SDE-Net alternates between drift net and diffuse net. 
The drift net increases the accuracy of the prediction and allows the measurement of the analistic entity, and the epistemical entity is measured with the diffusion net.

### There are two kinds of Uncertainties.

* Aleatoric Uncertainty

  - Respond to noise inherent in the data (such as distinguishing between 3 and 8).

  - Respond to randomness of data (such as tossing coins).

  - Data does not decrease even if there is more data.

  - Directly obtain the model's output.



* Epistemic Uncertainty

  - Responses should be made in areas the model does not know (unlearned data).

  - The more data you learn, the smaller the value becomes.

  - Obtain using sampling.
  
---------------------------------------------
## Neural ordinary differential equations(Chen et al.)

  
In order to understand the thesis, you must first know the Neural order differential values.
In this paper, a new optimize method was presented. The use of an orderary differential calculation is typical of the Euler method.

The Euler method is a method used to find unknown curves. This method assumes that you know the starting point and the differential equation of the corresponding curve (the slope at all points can be obtained).

EX) As shown in the figure below, assume that A0 is the starting point (initial value). The points of A1 can be obtained by multiplying the slope at A0 by Δt.
Assuming that the points obtained are above the curve, the same process can be repeated to obtain up to A4.
  
  ![image](https://user-images.githubusercontent.com/46440177/99496383-85793800-29b7-11eb-85fd-fb78a038c33f.png)
> https://en.wikipedia.org/wiki/Euler_method




This can be simply expressed as ![image](https://user-images.githubusercontent.com/46440177/99497163-de959b80-29b8-11eb-8af0-6d23b1ef3787.png).

if f(x,t) is replaced with dx/dt, it is ![image](https://user-images.githubusercontent.com/46440177/99497392-3502da00-29b9-11eb-8793-6bb5c23d76ac.png)

If you think of Neural Net as a dynamic system and think of each layer of NN as a continuous process, you can use ODE. 
If Resnet's residual block exists consecutively, it can be expressed as follows:  

![image](https://user-images.githubusercontent.com/46440177/99497614-890dbe80-29b9-11eb-9952-19d891d77500.png)

If the expression above is generalized, it is ![image](https://user-images.githubusercontent.com/46440177/99497747-b0648b80-29b9-11eb-9322-d32e78ac0957.png)
If Δx is 1 in the Euler method, it is consistent with the expression in the residual net above. This can be understood as a process of finding a single curve.

The paper used the Adjoint Sensitivity method as the ODE solver to obtain the slope, reduce and update the error. (See thesis for details)
>  https://arxiv.org/abs/1806.07366


### The advantages of this method are as follows:

1. Faster testing time than RNN, but slower training time.

2. Time series prediction is more accurate.

3. Open the realm of new optimizing method.

4. The slope calculation takes less memory.

---------------------------------------------
## Stochastic Differential Equation(SDE) & Brownian motion

ODE is deterministic and does not estimate uncertainty. Therefore, use one method (SDE) that is stochastic. 
In addition to that, add brownian motion term (a phenomenon in which small particles move irregularly in a liquid or gas) to obtain epistemic uncertainty.
I think the idea of adding brownian motioin term to get an epistemic certificate seems fresh.

In the Euler method, if t is expressed as Δt and Δt→0, the above expression of ResNet can be expressed as follows.
![image](https://user-images.githubusercontent.com/46440177/99499396-2ff35a00-29bc-11eb-8581-ba879e791947.png)  

This is the normal ODE expression, and the SDE expression with the addition of Brownian motion term is as follows.
![image](https://user-images.githubusercontent.com/46440177/99499460-4bf6fb80-29bc-11eb-830d-62f6f7418343.png)  
(f : drift net, g : diffusion net)  

f(x,t) is the goal to make a good prediction and g(x,t) is to know the uncertainty. Therefore, if there is sufficient training data and the empirical unity is low, 
the variance of Brownian motion will be low, and if there is a lack of training data, the variance of Brownian motion will be large.

-----------------------------------------------------------
## Euler-Maruyama

In principle, stochastic dynamics can be simulated with high-order numerical solver, but the input data of deep leading usually has high dimension, so the cost is amazing. 
Therefore, a method called Euler-Maruyama is used here in a fixed step size for efficient training.

This method is a generalization of the Euler method from the ordinal differential equation(ODE) to the static differential equation(SDE).
Assume stochastic differential equation : ![image](https://user-images.githubusercontent.com/46440177/99499812-cde72480-29bc-11eb-94ce-78e2ebdf6861.png)  
Initial condition X0=x0 W(t) attempts to resolve the SDE at a certain time interval in the Wiener process, [0, T]. 
The Euler-Maruyama approximation for solution X is then the Markov chain Y defined as follows:
![image](https://user-images.githubusercontent.com/46440177/99500007-10106600-29bd-11eb-9856-3982efda9da9.png)  
(Δt = T/N, Y0 = x0, ΔW : i.i.d normal random variables (mean:0,variance:Δt)  
 
 The above is represented in code and the simulation results are as follows.(See [Wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method))

```python
import numpy as np
import matplotlib.pyplot as plt

num_sims = 5  # Display five runs

t_init = 3
t_end = 7
N = 1000  # Compute 1000 grid points
dt = float(t_end - t_init) / N
y_init = 0

c_theta = 0.7
c_mu = 1.5
# if you increase sigma, diffusion is increases. 
c_sigma = 1


def mu(y, t):
    """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
    return c_theta * (c_mu - y)


def sigma(y, t):
    """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
    return c_sigma


def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


ts = np.arange(t_init, t_end + dt, dt)
ys = np.zeros(N + 1)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = (i - 1) * dt
        y = ys[i - 1]
        ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
    plt.plot(ts, ys)

plt.xlabel("time (s)")
h = plt.ylabel("y")
plt.show()
```
![image](https://user-images.githubusercontent.com/46440177/99500693-0dfad700-29be-11eb-8a65-0a65b616d9e2.png)


![image](https://user-images.githubusercontent.com/46440177/99500720-13f0b800-29be-11eb-82c5-40119b895ed4.png)

The following formula is used for actual training.  
![image](https://user-images.githubusercontent.com/46440177/99500785-2c60d280-29be-11eb-9c82-9fac17705c53.png)  
![image](https://user-images.githubusercontent.com/46440177/99500842-40a4cf80-29be-11eb-8a5b-4b0c0681ba40.png)

-----------------------------------------------------------------
## SDE-Net : Drift net(f) + Diffusion net(g)
![image](https://user-images.githubusercontent.com/46440177/99500877-4dc1be80-29be-11eb-9084-5a3492fc6d55.png)  
Drift Net f aims to learn good predictive accuracy. It is also aimed at measuring the Aletoric uncertainty. 
In the case of the regression task, print mean and variance and learn to the NLL (such as [Simple and Scalable Predictive Uncertificate Estimation Using Deep Ensemble](https://arxiv.org/abs/1612.01474)).
Classification outputs mean and learns with cross entropy error.

Diffusion Net g aims to obtain an epistemic uncertainty. In-distribution (ID) data should have a smaller variance of Brownian motion. 
So the system state is determined by the left term and the output variance should be small. 
On the other hand, out-of-distribution (OOD) data should have a large variation of Brownian motion and the system should be chatic. 
When learning, use a binary cross entropy error to learn to distinguish between fake(OOD) and true(ID).

The code below makes it easier to understand. The SDENet class shows that the out is repeated and updated (in the case not training_diffusion) by layer_depth (this uses the Euler-Maruyama method above). 
Here it can be seen that all other variables are fixed and the std value of random normal variables is determined by whether the diffuse term is high or low.

The diffusion term multiplies the sigma_max value (scaling value) to the value passed sigmoid, thus having a variance size of 0 to sigma_max.

```python
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
```

-------------------------------------------------------------------
## Objective function for training

![image](https://user-images.githubusercontent.com/46440177/99501530-3800c900-29bf-11eb-8715-06d2d88a7e0d.png)  
(L: loss function, P_train: distribution for training data, P_ood: OOD data, T: terminal time of the stochastic process)
 
 first term : Minimize Loss with training data and learn.
 second term : When training data is inserted, let the diffusion net output low diffusion.
 third term : When out of distribution data is inserted, let the diffusion net output high diffusion. (In actual learning, a fake label is given to minimize loss.)
 (out of distribution data is the value added by Gaussian noise to the original value.)
 
 
 -----------------------------------------------------------------
 ## Algorithm  

![image](https://user-images.githubusercontent.com/46440177/99501931-c70de100-29bf-11eb-8a9e-b10ee9693f01.png)  

1. Use training data to pass downsampling layer to obtain (1).

2. (1) Use the Euler-Maruyama method to pass through the N steps. (N: layer depth)

3. Pass through the Fully connected layer.

4. Save Loss and learn Down Sampling layer, Fully connected layer, and drift net.

5. Obtain out-of-distribution data and pass down sampling layer to obtain (2).

6. Pass (1) and (2) through the diffusion net to obtain the output.

7. Give each true label and fake label to learn the diffusion net with the binary contextropy. (Learning that diffusion net separates ID from OOD)


As shown above, take turns learning the drift net and the diffuse net.

In code, it is shown below. The first slope became too big, so the nan value came out and we treated the clip by norm.  
First, learn drift net, fc layer, down sampling layer.  
Then, update the diffusion layer by learning the in-distribution data and out-of-distribution data alternately.

```python
real_label = 0
fake_label = 1

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
```

----------------------------------------------------
## Conclusion



- A model that estimates the uncertificate using Stochastic Differential Equation and Brownian motion is presented.



- The model will be divided into drift net and diffuse net to study and take charge of acuracy and epistemic certificate, respectively.



- Use the Euler-Maruyama method to proceed with learning.



- The advantages of the model are as follows.

    1. Using only one model, the cost of learning is smaller (than the ensemble method).



    2. It can distinguish between aleatoric uncertainty and an epistemic uncertainty.



    3. It is efficient because there is no need to specify the prior distribution and no need to estimate the posterior distribution.



    4. Both classification and regression can be performed.



    5. OOD(out of distribution) detection, Misclassification Detection, Adversarial Sample Detection, Active Learning all show good performance.



The official code is in [Github](github.com/Lingkai-Kong/SDE-Net) with pytorch, so when I train it using a colab, I got a similar result as suggested in the paper.  

The file converted to Tensorflow was created separately and placed on the github, but the performance was different from that of Torch.  
I thought I had created the same structure (except for adjusting the leading rate), but I don't know if there's anything else.

------------------------------------------------------------------------------------------------------------
## Reference
> ["SDE-Net: Equipping Deep Neural Networks with Uncertainty Estimates code"](https://arxiv.org/abs/2008.10546) (2020, ICML) - Lingkai Kong et al.  
> https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method  
> ["Neural Ordinary Differential Equations"](https://arxiv.org/abs/1806.07366) - Chen et al.  
> https://en.wikipedia.org/wiki/Euler_method  
> https://github.com/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb  
> https://github.com/Lingkai-Kong/SDE-Net  
 










  
  
