import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class InvarianceDataset:
    """
    Handles loading, storing, and splitting datasets for invariance detection problems.
    """
    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            The full dataset including features and target.
        target_col : str
            Name of the column containing the target variable.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        self.df = df.copy()
        self.target_col = target_col
        self.features_df = self.df.drop(columns=[target_col])
        self.target_series = self.df[target_col]

    def to_csv(self, path: Path|str, index: bool = False):
        """
        Save dataset to a CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            The full dataset including features and target.
        path : str
            Path to the CSV file.
        """
        df = self.df.to_csv(path, index=index)

    @classmethod
    def from_csv(cls, path: Path|str, target_col: str):
        """
        Load dataset from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        target_col : str
            Name of the target column.
        """
        df = pd.read_csv(path)
        return cls(df, target_col)
    
    def to_excel(self, path: Path|str, index: bool = False):
        """
        Save dataset to a XLSX file.

        Parameters
        ----------
        df : pandas.DataFrame
            The full dataset including features and target.
        path : str
            Path to the CSV file.
        """
        self.df.to_excel(path, index=index)

    @classmethod
    def from_excel(cls, path: Path|str, target_col: str, **kwargs):
        """
        Load dataset from an Excel file.

        Parameters
        ----------
        path : str
            Path to the Excel file.
        target_col : str
            Name of the target column.
        kwargs : additional arguments for pandas.read_excel
        """
        df = pd.read_excel(path, **kwargs)
        return cls(df, target_col)

    def summary(self):
        """Return basic statistics of the dataset."""
        return self.df.describe()

    def head(self, n=5):
        """Return the first n rows of the dataset."""
        return self.df.head(n)

    def train_test_split(self, test_size:float=0.15, random_state:int=0):
        """
        Split the dataset into training and test sets.

        Parameters
        ----------
        test_size : float
            Fraction of data to reserve for testing.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
            Train/test features and targets.
        """
        return train_test_split(
            self.features_df.values,
            self.target_series.values,
            test_size=test_size,
            random_state=random_state
        )

    @property
    def feature_names(self):
        """List of feature names."""
        return list(self.features_df.columns)
