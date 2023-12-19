import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import EarlyStopping
import pandas as pd
from scipy import signal, linalg
import tqdm
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
# dataset creation 
Data = pd.read_excel('Newton.xlsx', index_col = 0)

Fg = np.array(Data.iloc[:,8]).astype('float64')
noise = np.random.normal(loc = 0.0, scale = 0.1, size = len(Fg))               # create gaussian noise
Fg = Fg*(1+noise)                                                              # add noise

Data = Data.drop(columns = ['Fg'])
Data.insert(8, 'Fg', Fg)

Data.to_excel('Newton Noise.xlsx')

'''

Data = pd.read_excel('Newton Noise.xlsx', index_col = 0)

# data preprocessing

train_df, test_df = train_test_split(Data, test_size=0.15, random_state=0)
X_train = train_df.iloc[:, :-1]
X_test = test_df[X_train.columns]
y_train = train_df.iloc[:, -1]
y_test = test_df.iloc[:, -1]



normalizer = tf.keras.layers.Normalization(axis=-1)             
normalizer.adapt(np.array(X_train))                                            # Let the normalization layer choose how to normalize data on the training set


def build_and_compile_model(norm):                                             # Define the model
    model = Sequential([
      norm,
      Dense(128, activation = 'relu'),
      Dense(64, activation = 'relu'),
      Dense(64, activation = 'relu'),
      Dense(32, activation = 'relu'),
      Dense(32, activation = 'relu'),
      Dense(16, activation = 'relu'),
      Dense(16, activation = 'relu'),
      Dense(8, activation = 'relu'),
    
      Dense(1, activation = 'linear')
  ])

    model.compile(loss = 'mean_absolute_error',
                optimizer = Adam(0.001))
    return model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)    # Use early stopping regularization
model = build_and_compile_model(normalizer)                                    # Build the model

#time
history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    verbose=2, epochs=500, callbacks=[es])


test_predictions = model.predict(X_test).flatten()
test_labels = y_test.values                                                    # True y over samples of the testing set

model.save('newton_model.tf')                                                  # save model for later use

r2 = r2_score(test_labels, test_predictions)
mae = mean_absolute_error(test_labels, test_predictions)                       
rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))              # goodness metrics

delta = max(test_labels) - min(test_labels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3), dpi = 190)
ax1.scatter(test_labels, test_predictions, c="crimson", alpha=0.2)
p1 = max(max(test_predictions), max(test_labels))
p2 = min(min(test_predictions), min(test_labels))
ax1.plot([p1, p2], [p1, p2], "b-")
ax1.annotate(
    "$R^2$ = %0.3f" % r2,
    xy=(0.02 * delta, 0.95 * delta),
    xytext=(0.02 * delta, 0.95 * delta),
)
ax1.annotate(
    "MAE = %0.2f N" % mae,
    xy=(0.02 * delta, 0.85 * delta),
    xytext=(0.02 * delta, 0.85 * delta),
)
ax1.annotate(
    "RMSE = %0.2f N" % rmse,
    xy=(0.02 * delta, 0.75 * delta),
    xytext=(0.02 * delta, 0.75 * delta),
)
ax1.set_xlabel(r"True $\overline{F_{\rm{g}}}$ (N)")
ax1.set_ylabel(r"Predicted $\overline{F_{\rm{g}}}$ (N)")
ax1.text(-0.3, 0.9, 'a', transform=ax1.transAxes, 
            size=12, weight='bold')

ax2.plot(history.history['loss'], label='loss')
ax2.plot(history.history['val_loss'], label='validation loss')
ax2.text(-0.3, 0.9, 'b', transform=ax2.transAxes, 
            size=12, weight='bold')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE (N)')
ax2.legend(frameon = False)
plt.subplots_adjust(wspace=0.4)

plt.savefig('model.svg', bbox_inches='tight')

plt.show()


#starting feature grouping analysis
new_model = tf.keras.models.load_model('newton_model.tf')                      # alternatively, load saved model

other_columns = Data.iloc[:,:-1].columns

def univec(a, b):
    B = np.zeros([1, 2])
    B[0][0] = a
    B[0][1] = b                                                                # build the matrix B in x0
    un_norm = linalg.null_space(B)                                             # ortho-normal basis in the null-space of B

    return un_norm


def delta(l, Data, i):                                                         # to pick random points in the interval [x_min, m_max]

    x_1_min = np.min(Data[l[i][0]])
    x_1_max = np.max(Data[l[i][0]])
    x_2_min = np.min(Data[l[i][1]])
    x_2_max = np.max(Data[l[i][1]])

    return(x_1_min, x_1_max, x_2_min, x_2_max)

l = list(combinations(list(other_columns), 2))                                 # get all possible combinations of the features

N = 20                                                                         # number of points evaluated per iteration
repeat = 3                                                                     # number of iterations

A = np.zeros([len(l), N])
B = np.zeros([len(l), N])


for i in tqdm.tqdm(range(len(l))):
    
    a = -0.5*np.ones(N)                                                        # initial guess for first iteration 
    b = 0.5*np.ones(N)
    
    for j in range(repeat):
        
        beta = np.array([np.mean(a), np.mean(b)])                              # initial guess for successive iterations
        
        a = np.ones(N)
        b = np.ones(N)
        
        for k in range(N):
            
            w1 = np.random.uniform(0, 1)                                       # pick two random numbers
            w2 = np.random.uniform(0, 1)
            
            tx = Data[other_columns].describe().loc['mean']
            delta_values = delta(l, Data, i)
            
            index_1 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(l[i][0])     # get index of the first feature
            index_2 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(l[i][1])     # get index of the second feature
    
            def MAIN(beta):
    
                a0 = beta[0]                                                   # initial guess
                b0 = beta[1]
                
                tx = Data[other_columns].describe().loc['mean']                # set feature values to their average
                r = delta_values[0] + w1*(delta_values[1] - delta_values[0])   # pick the value of the first feature at random in the interval x1_min, x1_max
                tx[l[i][0]] = r                                                # set the value of the x0 component corresponding to the first feature in the couple equal to r
                s = delta_values[2] + w2*(delta_values[3] - delta_values[2])
                tx[l[i][1]] = s
                x0 = tf.constant(tx)
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)                                                 
                    preds1 = new_model(x0)                                     # to differentiate the DNN model
                dy_dx1 = tape.gradient(preds1, x0)[index_1]                    # evaluate the partial derivative of the model in x0 with respect to the first feature
    
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)
                    preds2 = new_model(x0)
                dy_dx2 = tape.gradient(preds2, x0)[index_2]
    
    
                grad1 = dy_dx1.numpy()                                         # convert to array
                grad2 = dy_dx2.numpy()
    
                der = np.array([grad1, grad2])                                 # create derivative vector which has to be parallel to the orthogonal vector un in x0
                der = der/np.sqrt(der[0]**2 + der[1]**2)                       # normalization
    
                tx = Data[other_columns].describe().loc['mean']                # need to create x0 again, so set features to average
                tx[l[i][0]] = r                                                # set the value of the x0 component corresponding to the first feature in the couple equal to r
                tx[l[i][1]] = s                                                # set the value of the x0 component corresponding to the second feature in the couple equal to r
                x0 = tf.constant(tx)                                           # create x0
    
                un = univec(a0, b0)                                            # get un to check for the condition of invariance (components of the gradient aligned with un in x0)
                DIFF = np.array([np.dot(der, un),
                                 a0**2 + b0**2 - 1], dtype = float)            # evaluate the resiudal
    
                return(DIFF)
            
            
            x = least_squares(MAIN, beta, method = 'trf', ftol = 2.3e-16, 
                              verbose = 0, max_nfev = 50)                      # least-squares with Levenberg-Marquardt does not work with rectangular matrices
            x.x = x.x/np.linalg.norm(x.x)
            
            a[k] = x.x[0]                                                      # save the results for each of the N iterations
            b[k] = x.x[1]

    A[i] = a                                                                   # create matrix with the results, the ith row containing the found exponents for the first feature in the ith couple
    B[i] = b 
    
a_b = np.array([np.mean(A, axis = 1), 
                np.std(A, axis = 1), 
                np.mean(B, axis = 1), 
                np.std(B, axis = 1)]).transpose()                              # create "summary" containing mean and std of the results

a_b_frame = pd.DataFrame(a_b, columns = ['mean a', 'std a', 'mean b', 'std b'],
                         index = l)                                            # convert to dataframe for simplicity

# scatter plots
for ind in range(len(A)):
    plt.figure(figsize=(5, 3), dpi=190)
    plt.scatter(np.arange(N), A[ind])
    plt.scatter(np.arange(N), B[ind])
    plt.title(l[ind])
    plt.ylim(-1,1)
    plt.show()
    