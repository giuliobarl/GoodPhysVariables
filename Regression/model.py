import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path
import pickle

class InvarianceModel:
    """
    Wrapper for building, training, saving, and using a regression DNN for invariance detection.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_layers = (64, 32, 16)
            ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_layers : tuple
            Units in each hidden Dense layer.
        learning_rate : float
            Learning rate for Adam optimizer.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.model = None
        self.history = None

    def build_and_compile(
            self,
            X_train,
            activation: str = 'relu',
            learning_rate: float = 0.001,
            loss: str = 'mean_absolute_error'
            ):
        """Build and compile the Keras model."""
        self.learning_rate = learning_rate
        self.normalizer.adapt(X_train)
        layers = [self.normalizer]
        for units in self.hidden_layers:
            layers.append(tf.keras.layers.Dense(units, activation=activation))
        layers.append(tf.keras.layers.Dense(1, activation='linear'))

        self.model = tf.keras.Sequential(layers)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss=loss)

    def train(
            self,
            X_train,
            y_train,
            validation_split: float = 0.15,
            epochs: int = 200,
            patience: int = 20,
            checkpoint_path = None,
            verbose = 1,
            save_model: bool = True,
            model_path: str = None
            ):
        """Train the model with early stopping and optional checkpoint saving."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
                )]
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True
                    ))
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )
        if save_model:
            self.model.load_weights(checkpoint_path)
            self.model.save(model_path)

    def save_history(self, history_path: Path|str):
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)

    def load_history(self, history_path: Path|str):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return history

    def load_model(self, model_path: Path|str):
        self.model = tf.keras.models.load_model(model_path)

    def evaluate(self, X_test, y_test):
        """Return R², MAE, RMSE on test data."""
        preds = self.predict(X_test)
        return r2_score(y_test, preds), mean_absolute_error(y_test, preds), np.sqrt(mean_squared_error(y_test, preds))

    def predict(self, X):
        """Predict outputs for given features."""
        return self.model.predict(X).flatten()
