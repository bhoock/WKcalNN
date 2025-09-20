#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from copy import deepcopy
import time
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Box, Layout, GridBox
import numpy as np
from IPython.display import display, Image
import mpl_interactions.ipyplot as iplt
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)  # Redirect stderr (fd=2) to /dev/null
import tensorflow
import tensorflow.keras.config
from tqdm.keras import TqdmCallback
from tensorflow.keras import saving
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Multiply, Lambda, Flatten, Reshape, Cropping1D, Activation
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.utils import get_custom_objects


def MAE(aa,bb):
  if len(aa) != len(bb):
    raise TypeError('MAE: mismatch shapes y_true and y_pred.')
  else:
    return np.mean(np.abs(aa-bb))


# sawtooth activation
def saw_act(x):
    # xmin = -1.7244706801327625
    # xmax = 1.7395976159234017
    xmin = -1.7406771593586796
    xmax = 1.7315610147941023
    c = (xmax - xmin)
    y = (x-xmin)/c
    f = y - tensorflow.floor(y)
    return c * f + xmin

class BiasLayer(tensorflow.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize bias to zero
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        return inputs + self.bias

class AddDummyInputs(tensorflow.keras.layers.Layer):
    def __init__(self, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs

    def build(self, input_shape):
        self.constantau_inputs = self.add_weight(
            shape=(self.num_inputs,),
            initializer='zeros',  # or anything else
            trainable=True,
            name='dummy_inputs'
        )

    def call(self, inputs):
        batch_size = tensorflow.shape(inputs)[0]
        fixed = tensorflow.tile(tensorflow.expand_dims(self.constantau_inputs, 0), [batch_size, 1])
        return tensorflow.concat([inputs, fixed], axis=-1)

def sum_of_squares(y_true, y_pred):
    return tensorflow.keras.backend.sum(tensorflow.keras.backend.square (y_pred - y_true))

class StopAtLossThreshold(tensorflow.keras.callbacks.Callback):
    def __init__(self, threshold=1e-6):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        this_loss = logs.get("loss")
        if this_loss is not None and this_loss < self.threshold:
            print(f"\nEarly stopping: {this_loss:.2e} z threshold {self.threshold:.2e}")
            self.model.stop_training = True

def pressure_dim(p,pM,pS):
  mmHg = 133.322387415
  return (p*pS+pM)/mmHg

# Heuristic for initial resistance
def get_guess_R(diastole,systole,position):
  coeffs_R = np.array([0.05185583, 0.11721294, 0.00425853, 1.21191814])
  return np.array([diastole,systole,position,1.]) @ coeffs_R



### part of notebook

# setup
def set_up(b):
    # Read options
    Mn = Mn_dd.value
    for iz in range(len(set_z)):
        if z_slider.value/100. < set_z[iz]:
            break
        else:
           iz_target = iz
    random_start = phase_check.value
    add_noise = noise_check.value
    opt_z = (z_dd.value  == 'position unknown')

    # Create target
    if Mn == 1:
      ir_target,ic_target = (12,12)
    if Mn == 2:
      ir_target,ic_target = (21,5)
    if Mn == -1:
      ir_target,ic_target = np.random.randint(0,25,2)
    if opt_z:
      iz_target = np.random.randint(0,31)
    idx_in = iz_target*len(set_r)*len(set_c)*len(set_t) + ir_target*len(set_c)*len(set_t) + ic_target*len(set_t)
    idx_out = idx_in + len(set_t)
    pressures_target = (pressures[idx_in:idx_out]-pM)/pS
    if random_start:
      it_random = np.random.randint(0,228)
      pressures_target = np.roll(pressures_target,it_random)
    # amin,amax = np.argmin(pressures_target),np.argmax(pressures_target)

    if add_noise:
      nb = slider_mu.value
      ns = slider_sigma.value
      pressures_target_no_noise = deepcopy(pressures_target)
      pressures_target = pressures_target+np.random.normal(0,ns*mmHg/pS,size=(len(pressures_target)))+nb*mmHg/pS
      N_target = pressures_target.shape[0]

    minimum = np.array([set_r[ir_target], set_c[ic_target], set_z[iz_target]])
    if opt_z == False: print("True parameters: \t R = {0:.4f} \t C = {1:.4f}".format(minimum[0],minimum[1],minimum[2]*100.))
    if opt_z == True: print("True parameters: \t R = {0:.4f} \t C = {1:.4f} \t z = {2:2.2f}".format(minimum[0],minimum[1],minimum[2]*100.))
    globals().update(locals())
    return 0

# plot target
def plot_target(b):
    # Plot the target
    plt.plot(period*TS+TM,pressure_dim(pressures_target,pM,pS),"bo",label="measurement")
    plt.xlim(-0.05, 0.9)
    plt.ylim(60, 150)
    plt.grid(True)
    plt.xlabel("time $t$ [s]")
    plt.ylabel("pressure $p(t) [mmHg]$")
    plt.legend()
    plt.tight_layout()
    plt.show()

# combine two functions above
def set_up_and_show(b):
  with output:
    set_up(b)
    plot_target(b)

# set up model
def set_up_model(b):
    # Load weights
    thin = model_dd.value
    if thin == 0:
      tensorflow.keras.Model.load_weights(model,'models/model_all_data.weights.h5')
    if thin == "g":
      tensorflow.keras.Model.load_weights(model,'models/model_reduced_mesh.weights.h5')


    model2.get_layer('l1').set_weights(model.layers[0].get_weights())
    model2.get_layer('l2').set_weights(model.layers[1].get_weights())
    model2.get_layer('l3').set_weights(model.layers[2].get_weights())
    model2.get_layer('l4').set_weights(model.layers[3].get_weights())

    for l in model2.layers:
      l.trainable = False
    if opt_z: model2.get_layer('dz').trainable=True
    model2.get_layer('drc').trainable=True
    model2.get_layer('ltb').trainable=True

    uez = False
    if uez:
      uz = set_z[iz_target] + np.linspace(-0.02,0.02,21)[iuz]
      uz = np.clip(uz,set_z[0],set_z[-1])
      z_in = (uz-ZM)/ZS
    else:
      z_in = (set_z[iz_target]-ZM)/ZS
    if opt_z:
      iz_random = np.random.randint(0,31)
      z_in = (set_z[iz_random]-ZM)/ZS
    model2.get_layer('dz').set_weights([np.array([z_in])])

    p=model2.predict(period,verbose=0).flatten()
    tau_in = np.argmin(p)-np.argmin(pressures_target)
    tau_in = tau_in*time_increment
    tau_in = saw_act(tau_in).numpy()
    model2.get_layer('ltb').set_weights([np.array([tau_in])])

    if uez: r_in = get_guess_R(pressures_target.min(),pressures_target.max(),uz)
    if opt_z: r_in = get_guess_R(pressures_target.min(),pressures_target.max(),set_z[iz_random])
    else: r_in = get_guess_R(pressures_target.min(),pressures_target.max(),set_z[iz_target])
    r_in = (r_in*1e8-RM)/RS

    errors = []
    for c_in in np.linspace(0.8,1.1,31):
        model2.get_layer('drc').set_weights([np.array([r_in,(c_in*1e-8-CM)/CS])])
        p_temp=model2.predict(period,verbose=0).flatten()
        errors.append(np.mean((pressures_target-p_temp)**2))
    c_in = np.linspace(0.8,1.1,31)[np.argmin(errors)]
    c_in = (c_in*1e-8-CM)/CS
    model2.get_layer('drc').set_weights([np.array([r_in,c_in])])

    if opt_z == False: print("Initial values: \t R = {0:.4f} \t C = {1:.4f}\n\n".format((r_in*RS+RM)*1e-8,(c_in*CS+CM)*1e8))
    if opt_z == True: print("Initial values: \t R = {0:.4f} \t C = {1:.4f} \t z = {2:2.2f}\n\n".format((r_in*RS+RM)*1e-8,(c_in*CS+CM)*1e8,(z_in*ZS+ZM)*100.))
    # print("Initial values: \t R = {0:.4f} \t C = {1:.4f} \t z = {2:2.2f} \t t = {3:.4f}\n\n".format((r_in*RS+RM)*1e-8,(c_in*CS+CM)*1e8,(z_in*ZS+ZM)*100.,tau_in*TS))

# Calibration
def run_calibration(b):
    with output:
        verb = verbose_check.value
        if verb == False:
          verb = 0
        if verb == True:
          verb = 1
        if verb != 0: print('Calibrartion started.\n')
        callback_es1 = EarlyStopping(monitor='loss',patience=50,min_delta=1e-5,verbose=verb,mode='min',start_from_epoch=100)
        callback_es2 = EarlyStopping(monitor='loss',patience=50,min_delta=1e-6,verbose=verb,mode='min',start_from_epoch=300)

        model2.compile(optimizer=Adam(learning_rate=0.01), loss=sum_of_squares, metrics=['mae'])
        t1 = time.time()
        if verb != 0: print('Learning rate 0.01')
        if verb != 0: model2.fit(period,pressures_target,epochs=500,batch_size=None,verbose=0,callbacks=[callback_es1,StopAtLossThreshold(threshold=1e-5),TqdmCallback(verbose=0)])
        else: model2.fit(period,pressures_target,epochs=500,batch_size=None,verbose=0,callbacks=[callback_es1,StopAtLossThreshold(threshold=1e-5)])

        model2.compile(optimizer=Adam(learning_rate=0.001), loss=sum_of_squares, metrics=['mae'])
        if verb != 0: print('Learning rate 0.001')
        if verb != 0: model2.fit(period,pressures_target,epochs=500,batch_size=None,verbose=0,callbacks=[callback_es2,StopAtLossThreshold(threshold=1e-6),TqdmCallback(verbose=0)])
        else: model2.fit(period,pressures_target,epochs=500,batch_size=None,verbose=0,callbacks=[callback_es2,StopAtLossThreshold(threshold=1e-6)])
        t1 = time.time()-t1
        #print('Evaluation to get parameters in ms %i' %(int(t1 * 1000)))

        R_opt,C_opt = model2.get_layer('drc').get_weights()[0]
        R_opt = (R_opt*RS+RM)*1e-8
        C_opt = (C_opt*CS+CM)*1e8
        rels = [(R_opt-minimum[0])/minimum[0],(C_opt-minimum[1])/minimum[1]]
        tau_opt = model2.get_layer('ltb').get_weights()[0][0]*TS  # Delta!
        if opt_z:
          z_opt = model2.get_layer('dz').get_weights()[0][0]*ZS+ZM
          rels.append((z_opt-minimum[2])/minimum[2])
        rels=np.array(rels)*100.

        print("\n\nCalibration finished.\n")
        if not opt_z:
          print("Result of calibration: \t\t R = %.4f \t C = %.4f" % (np.round(R_opt,4),np.round(C_opt,4)))
          print("True parameters: \t\t R = %.4f \t C = %.4f" % (minimum[0],minimum[1]))
          print("Percentage Errors: \t\t R: %.4f %% \t C = %.4f %%" % (np.round(rels[0],4),np.round(rels[1],4)))
        else:
          print("Result of calibration: \t\t R = %.4f \t C = %.4f \t z = %2.2f" % (np.round(R_opt,4),np.round(C_opt,4),z_opt*100))
          print("True parameters: \t\t R = %.4f \t C = %.4f \t z = %2.2f" % (minimum[0],minimum[1],minimum[2]*100.))
          print("Percentage Errors: \t\t R: %.4f %% \t C = %.4f %% \t z = %.4f" % (np.round(rels[0],4),np.round(rels[1],4),np.round(rels[2],4)))

        #print("\n\n")
        #print("Found phase shift [s]: \t %.4f" %tau_opt)

        if opt_z==False: return([R_opt,C_opt,tau_opt])
        if opt_z==True: return([R_opt,C_opt,z_opt,tau_opt])

# calibrate
def calibrate(b):
  with output:
    set_up_model(b)
    x = run_calibration(b)
  return x

# plot after calibration
def plot_after_cal(b):
    with output:
        p = model2.predict(period,verbose=0)
        p = p.flatten()
        p = pressure_dim(p,pM,pS)
        pt = pressure_dim(pressures_target,pM,pS)
        mae_cal = MAE(p,pt)
        p = model2.predict((period2-TM)/TS,verbose=0)
        p = p.flatten()
        p = pressure_dim(p,pM,pS)

        f, ax = plt.subplots(1,1)
        ax.plot(period2,p,"go",label="NN after calibration")
        ax.plot(period*TS+TM,pt,"bo",label="measurement")
        ax.set_xlim(-1.8,3)
        ax.set_ylim(np.min([pt.min(),p.min()])-10., np.max([pt.max(),p.max()])+10.)
        ax.set_xlabel("time $t$ [s]")
        ax.set_ylabel("pressure $p(t) [mmHg]$")
        ax.add_artist(AnchoredText("MAE NN [mmHg]: {:.4f}".format(mae_cal), loc=2))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()




get_custom_objects().update({'saw_act': Activation(saw_act)})
np.random.seed(42)

ZM,ZS,TM,TS,RM,RS,CM,CS,pM,pS = list(np.load('../../Data/Training_data/standardizers_DD.npy'))
period = np.load('../../Data/Training_data/period228.npy')
period2 = np.linspace(-1.5,2.5,1000)
pressures = np.load('../../Data/Training_data/Training_Data_27_February_pressures_rolled.npy')

set_r = np.round(np.load('../../Data/Training_data/Training_Data_27_February_set_r.npy')*1e-8,4)
set_c = np.round(np.load('../../Data/Training_data/Training_Data_27_February_set_c.npy')*1e8,4)
set_z = np.round(np.load('../../Data/Training_data/Training_Data_27_February_set_z.npy'),4)#*ZS+ZM
set_t = np.round(np.load('../../Data/Training_data/Training_Data_27_February_set_t.npy'),7)

mmHg = 133.322387415
start_points = np.array([[set_r[0],set_c[0]],[set_r[-1],set_c[0]],[set_r[0],set_c[-1]],[set_r[-1],set_c[-1]]])
time_increment = period[1]-period[0]


# Load the NN
model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='linear'))
# model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
model.compile(optimizer=Adam(learning_rate=0.01), loss=sum_of_squares, metrics=['mae'])


# Extended Surrogate
inputt = Input(shape=(1,))
ltb = BiasLayer(1,name='ltb')(inputt)
ltp = Activation(saw_act,name='ltp')(ltb)
inputtz = AddDummyInputs(num_inputs=1,name="dz")(ltp)
inputtzrc = AddDummyInputs(num_inputs=2,name="drc")(inputtz)
inputztrc = tensorflow.keras.layers.Lambda(lambda x: tensorflow.gather(x, [1, 0, 2, 3], axis=-1))(inputtzrc)

l1 = Dense(32,activation='relu',name='l1')(inputztrc)
l2 = Dense(32,activation='relu',name='l2')(l1)
l3 = Dense(32,activation='tanh',name='l3')(l2)
l4 = Dense(1,activation='linear',name='l4')(l3)
model2 = Model(inputs=[inputt],outputs=l4)
model2.compile(optimizer=Adam(learning_rate=0.001), loss=sum_of_squares, metrics=['mae'])
