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
from scipy import linalg
import tqdm
from scipy.optimize import least_squares
from itertools import combinations, product
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# data preprocessing
Data = pd.read_excel('Gnielinski Noise.xlsx', index_col = 0)

train_df, test_df = train_test_split(Data, test_size=0.15, random_state=0)
X_train = train_df.iloc[:, :-1]
X_test = test_df[X_train.columns]
y_train = train_df.iloc[:, -1]
y_test = test_df.iloc[:, -1]

#Let the normalization layer choose how to normalize data on the training set

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

def build_and_compile_model(norm):                                             # Define the model
    model = Sequential([
      norm,
      Dense(128, activation = 'relu'),
      Dense(64, activation='relu'),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(32, activation='relu'),
      Dense(16, activation='relu'),
      Dense(16, activation='relu'),
      Dense(8, activation='relu'),
    
      Dense(1, activation = 'linear')
  ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# Callbacks
es = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   verbose = 1, patience = 50)

checkpoint_filepath = './tmp/checkpoint_gniel'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True)

model = build_and_compile_model(normalizer)                                    # Build the model

#time
history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    verbose=2, epochs=500,
    callbacks = [es, model_checkpoint_callback])

with open('GnielHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)                                      # Save the model history
    
# Load saved weights (best model)
model.load_weights(checkpoint_filepath)
model.save('gnielinski_model.tf')                                              # Save the weights of the best model for later use


with open('GnielHistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

test_predictions = model.predict(X_test).flatten()
test_labels = y_test.values                                                    # True y over samples of the testing set

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
    "MAE = %0.2f" % mae,
    xy=(0.02 * delta, 0.85 * delta),
    xytext=(0.02 * delta, 0.85 * delta),
)
ax1.annotate(
    "RMSE = %0.2f" % rmse,
    xy=(0.02 * delta, 0.75 * delta),
    xytext=(0.02 * delta, 0.75 * delta),
)
ax1.set_xlabel(r"True $\overline{\rm{Nu}}$")
ax1.set_ylabel(r"Predicted $\overline{\rm{Nu}}$")
ax1.text(-0.3, 0.9, 'a', transform=ax1.transAxes, 
            size=12, weight='bold')

ax2.plot(history['loss'], label='loss')
ax2.plot(history['val_loss'], label='validation loss')
ax2.set_yscale('log')
ax2.text(-0.35, 0.9, 'b', transform=ax2.transAxes, 
            size=12, weight='bold')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend(frameon = False)
plt.subplots_adjust(wspace=0.4)


plt.savefig('gniel_model.svg', bbox_inches='tight')

plt.show()


###############################################################################
#starting feature grouping analysis
model = tf.keras.models.load_model('gnielinski_model.tf')                      # load saved model

other_columns = Data.iloc[:, :-1].columns

def univec(a, b, c, d, e, x0, index_1, index_2, index_3, index_4, index_5):    # build the matrix B in x0
    B = np.zeros((2, 4))

    if index_4 == index_1:                                                     # checks to create the matrix B properly according to which feature is common to both the couples
        B[0][0] = a/x0[index_1]
        B[0][1] = b/x0[index_2]
        B[0][2] = c/x0[index_3]
        
        B[1][0] = d/x0[index_4]
        B[1][3] = e/x0[index_5]
    elif index_4 == index_2:
        B[0][0] = a/x0[index_1]
        B[0][1] = b/x0[index_2]
        B[0][2] = c/x0[index_3]
        
        B[1][1] = d/x0[index_4]
        B[1][3] = e/x0[index_5]
    elif index_4 == index_3:
        B[0][0] = a/x0[index_1]
        B[0][1] = b/x0[index_2]
        B[0][2] = c/x0[index_3]
        
        B[1][2] = d/x0[index_4]
        B[1][3] = e/x0[index_5]
    
    un = linalg.null_space(B)                                                  # ortho-normal basis in the null-space of B
    
    un1 = np.transpose(un)[0]
    un2 = np.transpose(un)[1]

    return un1, un2


def delta(lista, Data, i):                                                     # to pick random points in the interval [x_min, m_max]

    x_1_min = np.min(Data[lista[i][0][0]])
    x_1_max = np.max(Data[lista[i][0][0]])
    x_2_min = np.min(Data[lista[i][0][1]])
    x_2_max = np.max(Data[lista[i][0][1]])
    x_3_min = np.min(Data[lista[i][0][2]])
    x_3_max = np.max(Data[lista[i][0][2]])
    x_4_min = np.min(Data[lista[i][1][0]])
    x_4_max = np.max(Data[lista[i][1][0]])
    x_5_min = np.min(Data[lista[i][1][1]])
    x_5_max = np.max(Data[lista[i][1][1]])

    return(x_1_min, x_1_max, x_2_min, x_2_max, x_3_min, x_3_max, x_4_min,
           x_4_max, x_5_min, x_5_max)


l_triples = list(combinations(list(other_columns), 3))
l_couples = list(combinations(list(other_columns), 2))
ll = list(product(*[l_triples, l_couples]))
lista = []

for i in range(len(ll)):
    if ll[i][1][0] in ll[i][0] and ll[i][1][1] not in ll[i][0]:
        lista.append(ll[i])
        
    elif ll[i][1][1] in ll[i][0] and ll[i][1][0] not in ll[i][0]:
        lista.append([ll[i][0], ll[i][1][::-1]])                               # get all possible combinations of sets of triplet and couple of the features,
                                                                               # such that the common features is always the third of the triplet and the second of the couple
        

N = 20                                                                         # number of points evaluated per iteration
repeat = 3                                                                     # number of iterations

A = np.zeros([len(lista), N])
B = np.zeros([len(lista), N])
C = np.zeros([len(lista), N])
D = np.zeros([len(lista), N])
E = np.zeros([len(lista), N])

for i in tqdm.tqdm(range(len(lista))):
    
    a = 0.5*np.ones(N)                                                         # initial guess for first iteration
    b = 0.5*np.ones(N)
    c = -0.5*np.ones(N)
    d = 0.5*np.ones(N)
    e = -0.5*np.ones(N)


    for j in range(repeat):
   
        beta = np.array([np.mean(a), np.mean(b), np.mean(c), np.mean(d), 
                         np.mean(e)])                                          # initial guess for successive operations
        
        a = np.ones(N)
        b = np.ones(N)
        c = np.ones(N)
        d = np.ones(N)
        e = np.ones(N)
        
        for k in range(N):
            
            w1 = np.random.uniform(0, 1)                                       # pick two random numbers
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)
            w5 = np.random.uniform(0, 1)
            
            tx = Data[other_columns].describe().loc['mean']
            delta_values = delta(lista, Data, i)
            
            index_1 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(
                                       lista[i][0][0])                         # get index of the first feature
            index_2 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(
                                       lista[i][0][1])                         # get index of the second feature
            index_3 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(
                                       lista[i][0][2])                         # get index of the third feature
            index_4 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(
                                       lista[i][1][0])                         # get index of the fourth feature
            index_5 = pd.DataFrame(tx, columns = 
                                   other_columns).columns.get_loc(
                                       lista[i][1][1])                         # get index of the fifth feature
            
        
            if index_4 == index_1:
                w4 = w1
            elif index_4 == index_2:
                w4 = w2
            elif index_4 == index_3:
                w4 = w3
                
                    
            
            def MAIN(beta):
        
                a0 = beta[0]                                                   # initial guess
                b0 = beta[1]
                c0 = beta[2]
                d0 = beta[3]
                e0 = beta[4]
                                                      
               
                tx = Data[other_columns].describe().loc['mean']                # set feature values to their average
                r = delta_values[0] + w1*(delta_values[1] - delta_values[0])   # pick the value of the first feature at random in the interval x1_min, x1_max
                tx[lista[i][0][0]] = r                                         # set the value of the x0 component corresponding to the first feature in the couple equal to r
                s = delta_values[2] + w2*(delta_values[3] - delta_values[2])
                tx[lista[i][0][1]] = s
                t = delta_values[4] + w3*(delta_values[5] - delta_values[4])
                tx[lista[i][0][2]] = t
                u = delta_values[6] + w4*(delta_values[7] - delta_values[6])
                tx[lista[i][1][0]] = u
                v = delta_values[8] + w5*(delta_values[9] - delta_values[8])
                tx[lista[i][1][1]] = v
                x0 = tf.constant(tx)                                           # create x0
                
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)                                                 
                    preds1 = model(x0)                                         # to differentiate the function
                dy_dx1 = tape.gradient(preds1, x0)[index_1]                    # evaluate the partial derivative of the model in x0 with respect to the first feature
        
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)
                    preds2 = model(x0)
                dy_dx2 = tape.gradient(preds2, x0)[index_2]
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)
                    preds3 = model(x0)
                dy_dx3 = tape.gradient(preds3, x0)[index_3]
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)
                    preds4 = model(x0)
                dy_dx4 = tape.gradient(preds4, x0)[index_4]
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x0)
                    preds5 = model(x0)
                dy_dx5 = tape.gradient(preds5, x0)[index_5]
        
                grad1 = dy_dx1.numpy()                                         # convert to array
                grad2 = dy_dx2.numpy()
                grad3 = dy_dx3.numpy()
                grad4 = dy_dx4.numpy()       
                grad5 = dy_dx5.numpy()
        
    
                der = np.array([grad1, grad2, grad3, grad5])                   # create derivative vector which has to be parallel to the orthogonal vector un in x0
                
                der = der/np.sqrt(
                    der[0]**2 + der[1]**2 + der[2]**2 + der[3]**2)             # normalization
        
                un1, un2 = univec(a0, b0, c0, d0, e0, x0.numpy(),
                                  index_1, index_2, index_3, index_4, index_5) # get un to check for the condition of invariance (components of the gradient aligned with un in x0)
                
                DIFF = np.array([np.dot(der, un1),
                                 np.dot(der, un2),
                                 a0**2 + b0**2 + c0**2 - 1, 
                                 d0**2 + e0**2 - 1],
                                dtype = float)                                 # evaluate the resiudal
        
                return(DIFF)
        
            x = least_squares(MAIN, beta, method = 'trf', ftol = 2.3e-16, 
                              verbose = 2, max_nfev = 20)                      # least-squares for rectangular matrices
            
            a[k] = x.x[0]
            b[k] = x.x[1]
            c[k] = x.x[2]
            d[k] = x.x[3]
            e[k] = x.x[4]

    A[i] = a
    B[i] = b
    C[i] = c
    D[i] = d
    E[i] = e

array = np.array([np.mean(A, axis = 1), 
                np.std(A, axis = 1), 
                np.mean(B, axis = 1), 
                np.std(B, axis = 1),
                np.mean(C, axis = 1),
                np.std(C, axis = 1),
                np.mean(D, axis = 1),
                np.std(D, axis = 1),
                np.mean(E, axis = 1),
                np.std(E, axis = 1)]).transpose()                              # create "summary" containing mean and std of the results

frame = pd.DataFrame(array, columns = ['mean a', 'std a', 'mean b', 'std b',
                                         'mean c', 'std c', 'mean d',
                                         'std d', 'mean e', 'std e'],
                     index = lista)                                            # convert to dataframe for simplicity

# scatter plots
for ind in range(len(A)):
    plt.figure(figsize=(6, 3), dpi=300)
    plt.scatter(np.arange(N), A[ind], label = r'$\alpha$')
    plt.scatter(np.arange(N), B[ind], label = r'$\beta$')
    plt.scatter(np.arange(N), C[ind], label = r'$\gamma$')
    plt.scatter(np.arange(N), D[ind], label = r'$\delta$')
    plt.scatter(np.arange(N), E[ind], label = r'$\epsilon')
    #plt.title(lista[ind])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.ylim(-1,1)
    plt.show()
