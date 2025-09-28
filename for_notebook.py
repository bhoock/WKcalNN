#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from copy import deepcopy
from for_models import *
import time
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Box, Layout, GridBox
import numpy as np
from IPython.display import display, Image
import mpl_interactions.ipyplot as iplt
import tkinter as tk
from tkinter import font as tkfont
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)  # Redirect stderr (fd=2) to /dev/null
import tensorflow
import tensorflow.keras.config
from tqdm.keras import TqdmCallback
from tensorflow.keras import saving
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Multiply, Lambda, Flatten, Reshape, Cropping1D, Activation
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.utils import get_custom_objects
plt.rcParams.update({'font.size': 14})


def MAE(aa,bb):
  if len(aa) != len(bb):
    raise TypeError('MAE: mismatch shapes y_true and y_pred.')
  else:
    return np.mean(np.abs(aa-bb))


# sawtooth activation
def saw_act(x):
    xmin = -1.7406771593586796
    xmax = 1.7315610147941023
    c = (xmax - xmin)
    y = (x-xmin)/c
    f = y - tensorflow.floor(y)
    return c * f + xmin

# bias dummy neuron
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

# add constant dummy neurons
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

# Speed up the backpropagation for training
@tensorflow.function  # compile the training to static graph
def train_step(x, y):
    with tensorflow.GradientTape() as tape: # use tape for autodiff
        pred = model2(x, training=True) # prediction by model
        loss_value = MSE_fn(y, pred)  # compute loss (MSE, defined outside)
    grads = tape.gradient(loss_value, model2.trainable_weights) # calculate gradients
    optimizer.apply_gradients(zip(grads, model2.trainable_weights)) # update the weights in optimizer
    return loss_value

def sum_of_squares(y_true, y_pred):
    return tensorflow.keras.backend.sum(tensorflow.keras.backend.square (y_pred - y_true))

MSE_fn = MeanSquaredError()
MAE_fn = MeanAbsoluteError()
optimizer = Adam(learning_rate=0.01)
optimizer_for_reset = Adam(learning_rate=0.01)

@tensorflow.function
def graph_pred(X):
  return model2(X,training=False)

@tensorflow.function
def graph_pred_NN(X):
  return model(X,training=False)

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


### part of notebook

# initialization
def initialize_NN(b):
  with output:
    # output.clear_output()
    verb = verbose_check.value
    if verb == False:
      verb = 0
    if verb == True:
      verb = 1
    make_plots = plots_check.value

    # Default values R, C
    r_in = model2.get_layer('d3').get_weights()[0][0]
    c_in = model2.get_layer('d4').get_weights()[0][0]
    # Default or known value z
    if opt_z == False:
      z_in = (set_z[iz_target]-ZM)/ZS
    if opt_z:
      iz_in = 15
      z_in = (set_z[iz_in]-ZM)/ZS
    model2.get_layer('d2').set_weights([np.array([z_in])])
    # Default value tau
    tau_in = model2.get_layer('d1a').get_weights()[0][0]
    if verb!=0: print("Default values:\t\t\t R = %.4f \t C = %.4f \t z = %2.2f \t tau = %.4f\n" % (np.round((r_in*RS+RM)*1e-8,4),np.round((c_in*CS+CM)*1e8,4),np.round((z_in*ZS+ZM)*100,4),np.round(tau_in*TS,4)))
    if make_plots: plot_ref_pred(b,label_nn='NN',small=True,no_NN=False,title="default values")

    # Get first guess tau
    p_temp=tensorflow.squeeze(graph_pred(period_b))
    dt_pred_t0 = period[np.argmin(p_temp)]-(-TM/TS)  # offset minium pred and t=0
    dt_target_t0 = period[np.argmin(pressures_target)]-(-TM/TS)  # offset minium target and t=0
    tau_in = (dt_pred_t0 - dt_target_t0)
    model2.get_layer('d1a').set_weights([np.array([tau_in])])
    if verb!=0: print("Initial guess time shift:\t tau = %.4f\n" %(tau_in*TS))
    if make_plots: plot_ref_pred(b,label_nn='NN',small=True,no_NN=False,title="initial guess for shift")

    # Initial values R and C (grid search)
    temp_errors = []
    range_params = np.linspace(-2,2,11)
    RR, CC = np.meshgrid(range_params,range_params)
    RRf = RR.flatten()
    CCf = CC.flatten()
    for r_in,c_in in zip(RRf,CCf):
      model2.get_layer('d3').set_weights([np.array([r_in])])
      model2.get_layer('d4').set_weights([np.array([c_in])])
      p_temp=graph_pred(period_b)
      temp_errors.append(MSE_fn(pressures_b,p_temp))
    temp_errors = np.array(temp_errors).reshape(11,11)
    flat_index = np.argmin(temp_errors)
    row, col = np.unravel_index(flat_index, temp_errors.shape)
    model2.get_layer('d3').set_weights([np.array([range_params[col]])])
    model2.get_layer('d4').set_weights([np.array([range_params[row]])])
    r_in = model2.get_layer('d3').get_weights()[0][0]
    c_in = model2.get_layer('d4').get_weights()[0][0]

    # Initial value z (grid search)
    if opt_z:
      temp_errors = []
      for z_in in range_params:
        model2.get_layer('d1a').set_weights([np.array([0.])])
        model2.get_layer('d2').set_weights([np.array([z_in])])
        p_temp=tensorflow.squeeze(graph_pred(period_b))

        # Adapt tau
        dt_pred_t0 = period[np.argmin(p_temp)]-(-TM/TS)  # offset minium pred and t=0
        dt_target_t0 = period[np.argmin(pressures_target)]-(-TM/TS)  # offset minium target and t=0
        tau_in = (dt_pred_t0 - dt_target_t0)
        model2.get_layer('d1a').set_weights([np.array([tau_in])])

        p_temp=tensorflow.squeeze(graph_pred(period_b))
        temp_errors.append(MSE_fn(pressures_b,p_temp))
      z_in = range_params[np.argmin(temp_errors)]
      model2.get_layer('d1a').set_weights([np.array([0.])])
      model2.get_layer('d2').set_weights([np.array([z_in])])
      p_temp=tensorflow.squeeze(graph_pred(period_b))
      tau_in = np.argmin(p_temp)-np.argmin(pressures_target)
      tau_in = tau_in*time_increment
      tau_in = saw_act(tau_in).numpy()
      model2.get_layer('d1a').set_weights([np.array([tau_in])])

    if verb!=0:
      if opt_z: print("Initial values:\t\t\t R = %.4f \t C = %.4f \t z = %2.2f" % (np.round((r_in*RS+RM)*1e-8,4),np.round((c_in*CS+CM)*1e8,4),np.round((z_in*ZS+ZM)*100,4)))
      else: print("Initial values:\t\t\t R = %.4f \t C = %.4f" % (np.round((r_in*RS+RM)*1e-8,4),np.round((c_in*CS+CM)*1e8,4)))
    if verb!=0: print("Initial value time shift:\t tau = %1.4f\n\n" % (np.round(tau_in*TS,4)))
    if make_plots: plot_ref_pred(b,label_nn='NN intialized',small=True,no_NN=False,title="after initialization")


# setup
def set_up(b):
    # Read options
    Mn = Mn_dd.value
    for iz in range(len(set_z)):
        if slider_z.value/100. < set_z[iz]:
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

    if add_noise:
      nb = slider_mu.value
      ns = slider_sigma.value
      pressures_target_no_noise = deepcopy(pressures_target)
      pressures_target = pressures_target+np.random.normal(0,ns*mmHg/pS,size=(len(pressures_target)))+nb*mmHg/pS
      N_target = pressures_target.shape[0]
    pressures_b = tensorflow.convert_to_tensor(pressures_target[:,None])

    minimum = np.array([set_r[ir_target], set_c[ic_target], set_z[iz_target]])
    if opt_z == False: print("True parameters: \t R = {0:.4f} \t C = {1:.4f}".format(minimum[0],minimum[1],minimum[2]*100.))
    if opt_z == True: print("True parameters: \t R = {0:.4f} \t C = {1:.4f} \t z = {2:2.2f}".format(minimum[0],minimum[1],minimum[2]*100.))
    globals().update(locals())
    return 0

# combine two functions above
def set_up_and_show(b):
  with output:
    output.clear_output()
    set_up(b)
    plot_ref_pred(b,no_NN=True,title="measurement"+" at R = {0:.4f}, C = {1:.4f}, z = {2:2.2f}".format(minimum[0],minimum[1],minimum[2]*100.))

# set up model
def set_up_model(b):
  with output:
    # output.clear_output()
    verb = verbose_check.value
    if verb == False:
      verb = 0
    if verb == True:
      verb = 1

    # Load weights
    if model_dd.value in ["0","G"]:
      thin = model_dd.value
    else:
      thin = custom_model_input.value
    try:
      assert thin in models_dict.keys()
    except:
      print("Invalid input for model!")
    tensorflow.keras.Model.load_weights(model,'models/'+models_dict[thin])
    if verb != 0:
      if thin=="0": print('Loaded model: all data\n')
      else: print('Loaded model: %s\n' %thin)


    model2.get_layer('l1').set_weights(model.layers[0].get_weights())
    model2.get_layer('l2').set_weights(model.layers[1].get_weights())
    model2.get_layer('l3').set_weights(model.layers[2].get_weights())
    model2.get_layer('l4').set_weights(model.layers[3].get_weights())
    model2.get_layer('d3').set_weights([np.array([0.0])])
    model2.get_layer('d4').set_weights([np.array([0.0])])
    model2.get_layer('d1a').set_weights([np.array([0.0])])
    model2.get_layer('d2').set_weights([np.array([0.0])])
    for l in model2.layers:
      l.trainable = False
    if opt_z: model2.get_layer('d2').trainable=True
    model2.get_layer('d3').trainable=True
    model2.get_layer('d4').trainable=True
    model2.get_layer('d1a').trainable=True


# Calibration
def run_calibration(b):
    with output:
        # output.clear_output()
        verb = verbose_check.value
        if verb == False:
          verb = 0
        if verb == True:
          verb = 1
        if verb != 0: print('Calibrartion started.\n')
        make_plots = plots_check.value
        es = es_check.value

        # do intialization
        initialize_NN(b)

        # Calibration itself (new)
        N_epochs = 2000
        optimizer=Adam.from_config(optimizer_for_reset.get_config())
        model2.optimizer.learning_rate.assign(0.01)
        if verb != 0: print('Learning rate 1: %.5f' %(np.round(np.double(model2.optimizer.learning_rate.value),5)))

        losses = np.zeros(N_epochs)
        params = np.zeros((N_epochs,4))

        t1 = time.time()
        for epoch in range(N_epochs):
          if epoch == 500:
            model2.optimizer.learning_rate.assign(0.001)
            if verb != 0: print('Learning rate 2: %.5f (switched at epoch %i)' %(np.round(np.double(model2.optimizer.learning_rate.value),5),epoch+1))
          losses[epoch]=train_step(period_b,pressures_b)
          if es and epoch>=200:
            if epoch % 50 == 0:
              if np.mean(np.abs(np.diff(losses[epoch-50:epoch])))<=1e-8:
                if verb != 0: print('Early stopped at epoch %i.' %(epoch+1))
                break
        t1 = time.time()-t1
        if verb != 0: print('Elapsed time to get parameters (ms) %i' %(int(t1 * 1000)))

        # Plot losses
        if make_plots:
          fig, ax = plt.subplots(1,1,figsize=(7,4))
          ax.plot(np.log10(losses),"ro")
          ax.set_xlabel("training epochs")
          ax.set_ylabel("$\log \hat{J}$")
          plt.setp(ax.spines.values(), lw=1.5)
          plt.grid(True)
          plt.title('Calibration Loss (log MSE)')
          plt.tight_layout()
          plt.show()


        R_opt = model2.get_layer('d3').get_weights()[0][0]
        C_opt = model2.get_layer('d4').get_weights()[0][0]
        R_opt = (R_opt*RS+RM)*1e-8
        C_opt = (C_opt*CS+CM)*1e8
        tau_opt = model2.get_layer('d1a').get_weights()[0][0]  # Delta!
        if opt_z: z_opt = model2.get_layer('d2').get_weights()[0][0]*ZS+ZM

        rels = [(R_opt-minimum[0])/minimum[0],(C_opt-minimum[1])/minimum[1]]
        # if opt_z:
        #     rels.append((z_opt-minimum[2])/minimum[2])
        rels=np.array(rels)*100.

        if verb != 0: print("\n\nCalibration finished.\n")
        if not opt_z:
          print("Result of calibration: \t\t R = %.4f \t C = %.4f" % (np.round(R_opt,4),np.round(C_opt,4)))
          print("True parameters: \t\t R = %.4f \t C = %.4f" % (minimum[0],minimum[1]))
          print("Percentage errors: \t\t R: %.4f %% \t C = %.4f %%" % (np.round(rels[0],4),np.round(rels[1],4)))
        else:
          print("Result of calibration: \t\t R = %.4f \t C = %.4f \t z = %2.2f cm" % (np.round(R_opt,4),np.round(C_opt,4),np.round(z_opt*100,2)))
          print("True parameters: \t\t R = %.4f \t C = %.4f \t z = %2.2f cm" % (minimum[0],minimum[1],minimum[2]*100.))
          print("Percentage errors: \t\t R: %.4f %% \t C = %.4f %%" % (np.round(rels[0],4),np.round(rels[1],4)))
        print("Found phase shift: \t\t tau = %.4f s" %(tau_opt*TS))
        print("============================================\n\n\n")

        # Formatting for popup message
        R = np.round(R_opt, 4)
        C = np.round(C_opt, 4)
        if opt_z: z = np.round(z_opt*100,2)
        tau = np.round(tau_opt*TS,4)
        R_true = minimum[0]
        C_true = minimum[1]
        if opt_z: z_true = minimum[2]
        R_err = np.round(rels[0], 3)
        C_err = np.round(rels[1], 3)

        if not opt_z:
          message = f"""          Result of calibration:  R = {R:.4f}    C = {C:.4f}\n
          True parameters:  R = {R_true:.4f}    C = {C_true:.4f}\n
          Percentage errors:  R: {R_err:.3f}%    C = {C_err:.3f}%\n
          Found phase shift tau:  {tau:.4f} s"""
        else:
          message = f"""          Result of calibration:  R = {R:.4f}    C = {C:.4f}    z = {z:2.2f} cm\n
          True parameters:  R = {R_true:.4f}    C = {C_true:.4f}    z = {z_true:2.2f} cm\n
          Percentage errors:  R: {R_err:.3f}%    C = {C_err:.3f}%\n
          Found phase shift tau:  {tau:.4f} s"""

        show_popup(message,"Calibration Result")

        if opt_z==False: return([R_opt,C_opt,tau_opt*TS])
        if opt_z==True: return([R_opt,C_opt,z_opt,tau_opt*TS])

# calibrate
def calibrate(b):
  with output:
    # output.clear_output()
    set_up_model(b)
    x = run_calibration(b)
  return x

# Plot measurement and NN
def plot_ref_pred(b,**kwargs):
  with output:
    # output.clear_output()
    if "label_NN" in kwargs:
      label_nn = kwargs["label_NN"]
    else:
      label_nn = "NN"
    if "small" in kwargs:
      small = kwargs["small"]
    else:
      small = True
    if "no_NN" in kwargs:
      no_NN = kwargs["no_NN"]
    else:
      no_NN = True
    if "title" in kwargs:
      title = kwargs["title"]
    else:
      title = ""
    pt = pressure_dim(pressures_target,pM,pS) # target
    if no_NN==False:
      pm = model2.predict(period,verbose=0) # model
      pm = pm.flatten()
      pm = pressure_dim(pm,pM,pS)
      mae_cal = MAE_fn(pm,pt)
      if small==False:
        pm = model2.predict((period2-TM)/TS,verbose=0)
        pm = pm.flatten()
        pm = pressure_dim(pm,pM,pS)

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.plot(period*TS+TM,pt,"b+",label="measurement")
    if small==False: ax.set_xlim(-1.8,3)
    if no_NN==False:
      if small:
        ax.plot(period*TS+TM,pm,"g--",label=label_nn)
      else:
        ax.plot(period2,pm,"g--",label=label_nn)
    if no_NN:
      ax.set_ylim(np.min([pt.min()])-15., np.max([pt.max()])+30.)
    else:
      ax.set_ylim(np.min([pt.min(),pm.min()])-15., np.max([pt.max(),pm.max()])+30.)
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel("pressure $p(t)$ [mmHg]")
    if no_NN==False:
      ax.add_artist(AnchoredText("MAE [mmHg]: {:.4f}".format(mae_cal), loc=2))
    plt.setp(ax.spines.values(), lw=1.5)
    plt.grid(True)
    plt.legend(loc='upper right',ncol=2)
    if title != "": plt.title(title)
    plt.tight_layout()
    plt.show()


def show_popup(message,title):
  root = tk.Tk()
  root.title(title)

  # Set window size and position
  window_width = 500
  window_height = 200
  screen_width = root.winfo_screenwidth()
  screen_height = root.winfo_screenheight()
  x_pos = int((screen_width - window_width) / 2)
  y_pos = int((screen_height - window_height) / 2)
  root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

  # Use a monospace font for alignment
  mono_font = tkfont.Font(family="Courier", size=11)

  # Create a Text widget for better formatting
  text = tk.Text(root, wrap="none", font=mono_font, bg="white", relief="flat")
  text.insert("1.0", message)
  text.config(state="disabled")  # Make it read-only
  text.pack(padx=20, pady=20, expand=True, fill='both')

  # Add a Close button
  close_button = tk.Button(root, text="Close", command=root.destroy)
  close_button.pack(pady=(0, 10))

  root.mainloop()


def show_popup(message,title):
    # Hidden root window
    root = tk.Tk()
    root.withdraw()

    # Create the top-level pop-up window
    popup = tk.Toplevel()
    popup.title(title)
    popup.configure(bg="#f9f9f9")

    # Set dynamic size and position
    popup.geometry("520x220")  # You can increase height if needed
    popup.resizable(True, False)

    # Use a nice sans-serif font
    nice_font = tkfont.Font(family="Helvetica", size=14)

    # Frame for padding and better layout
    frame = tk.Frame(popup, bg="#f9f9f9")
    frame.pack(expand=True, fill='both', padx=20, pady=15)

    # Use a Label widget for clean, auto-wrapping text
    label = tk.Label(
        frame,
        text=message,
        font=nice_font,
        justify="left",
        bg="#f9f9f9",
        anchor="w"
    )
    label.pack(expand=True, fill='both')

    # Close button
    close_btn = tk.Button(popup, text="Close", command=popup.destroy, font=nice_font)
    close_btn.pack(pady=(0, 10))

    # Focus and raise the window
    popup.lift()
    popup.attributes('-topmost', True)
    popup.after_idle(popup.attributes, '-topmost', False)


get_custom_objects().update({'saw_act': Activation(saw_act)})
np.random.seed(42)

# path_data = '../../Data/Training_data/'
path_data = os.getcwd()+'/Data/'
ZM,ZS,TM,TS,RM,RS,CM,CS,pM,pS = list(np.load(path_data+'standardizers_DD.npy'))
period = np.load(path_data+'period228.npy')
period_b = tensorflow.convert_to_tensor(period[:,None])
period2 = np.linspace(-1.5,2.5,1000)
period2_b = tensorflow.convert_to_tensor(period2[:,None])
# pressures = np.load(path_data+'Training_Data_27_February_pressures_rolled.npy')
pressures = np.r_[np.load(path_data+'Training_Data_27_February_pressures_rolled_1.npy'),np.load(path_data+'Training_Data_27_February_pressures_rolled.npy')]

set_r = np.round(np.load(path_data+'Training_Data_27_February_set_r.npy')*1e-8,4)
set_c = np.round(np.load(path_data+'Training_Data_27_February_set_c.npy')*1e8,4)
set_z = np.round(np.load(path_data+'Training_Data_27_February_set_z.npy'),4)#*ZS+ZM
set_t = np.round(np.load(path_data+'Training_Data_27_February_set_t.npy'),7)

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
model.compile(optimizer=optimizer, loss=MSE_fn, metrics=['mae'])


# Extended Surrogate
inputt = Input(shape=(1,))
d1a = BiasLayer(1,name='d1a')(inputt)
d1b = Activation(saw_act,name='d1b')(d1a)
d2 = AddDummyInputs(num_inputs=1,name='d2')(d1b)
d3 = AddDummyInputs(num_inputs=1,name='d3')(d2)
d4 = AddDummyInputs(num_inputs=1,name='d4')(d3)
inputztrc = tensorflow.keras.layers.Lambda(lambda x: tensorflow.gather(x, [1, 0, 2, 3], axis=-1))(d4)

l1 = Dense(32,activation='relu',name='l1')(inputztrc)
l2 = Dense(32,activation='relu',name='l2')(l1)
l3 = Dense(32,activation='tanh',name='l3')(l2)
l4 = Dense(1,activation='linear',name='l4')(l3)
model2 = Model(inputs=[inputt],outputs=l4)
model2.compile(optimizer=optimizer, loss=MSE_fn, metrics=['mae'])
