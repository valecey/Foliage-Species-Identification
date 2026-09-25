"""
SVM training module with biased classification support.
"""
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import resample
from typing import Optional, Dict, List, Tuple, Callable


class SVMTrainer:
    """Trains SVM models with optional bias toward sensitivity."""
    
    def __init__(
        self,
        gamma_values: List[float],
        c_values: List[float],
        use_biased_svm: bool = False,
        w_c_values: Optional[List[float]] = None,
        sensitivity_target: float = 0.95,
        n_splits: int = 3,
        verbosity: int = 1,
        n_jobs: int = 1,
        random_state: int = 42
    ):
        """
        Initialize SVM trainer.
        
        Args:
            gamma_values: Gamma values for RBF kernel
            c_values: C values (regularization)
            use_biased_svm: Whether to use biased scoring
            w_c_values: Class weight values for biased SVM
            sensitivity_target: Target sensitivity for biased SVM
            n_splits: Number of cross-validation splits
            verbosity: Verbosity level
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.gamma_values = gamma_values
        self.c_values = c_values
        self.alpha_values = [1/c for c in c_values]
        self.use_biased_svm = use_biased_svm
        self.w_c_values = w_c_values or []
        self.class_weights = [{0: w, 1: 1-w} for w in self.w_c_values]
        self.sensitivity_target = sensitivity_target
        self.n_splits = n_splits
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.best_model = None
    
    @staticmethod
    def biased_svm_score(clf, X, y) -> float:
        """
        Custom scoring function for biased SVM.
        Optimizes for sensitivity while penalizing false positives.
        Matches the original implementation exactly.
        
        Args:
            clf: Trained classifier
            X: Feature matrix
            y: True labels
            
        Returns:
            Score value (higher is better)
        """
        y_pred = clf.predict(X)
        # fragile, since we know CAMPAU is seen first by the labelencoder
        labels = [1, 0]
        confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred, labels=labels)
        tn, fp, fn, tp = confusion_matrix.ravel()
        
        # Avoid division by zero - exact match to original
        if fn == 0:
            fn = 1e-10
        if fp == 0:
            fp = 1e-10
        if tn == 0:
            tn = 1e-10
        if tp == 0:
            tp = 1e-10
        
        r = tp / (tp + fn)
        p_fx_is_1 = (tp + fp) / (tn + fn)
        
        return r**2 / p_fx_is_1
    
    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        species_name: str,
        negative_class_name: str = 'Not Selected'
    ) -> Tuple[Optional[object], LabelEncoder, Pipeline]:
        """
        Train SVM model with hyperparameter optimization.
        
        Args:
            train_X: Training features
            train_y: Training labels (string labels)
            test_X: Test features
            test_y: Test labels (string labels)
            species_name: Name of positive class
            negative_class_name: Name of negative class
            
        Returns:
            Tuple of (scaler, label_encoder, trained_model)
        """
        # Set up label encoder and transform string labels to numeric
        if not self.use_biased_svm:
            self.label_encoder.classes_ = np.asarray([species_name, 'Not Selected'])
        else:
            self.label_encoder.classes_ = np.asarray([species_name, negative_class_name])
        
        # Convert string labels to numeric
        train_y_numeric = self.label_encoder.transform(train_y)
        test_y_numeric = self.label_encoder.transform(test_y)
        
        # Phase 1: Find optimal gamma and C
        best_params = self._optimize_hyperparameters(train_X, train_y_numeric)
        
        print(f"param ln(C): {np.log(1/best_params['sgd__alpha'])}, "
              f"param ln(gamma): {np.log(best_params['rbf__gamma'])}")
        
        # For non-biased SVM, we're done
        if not self.use_biased_svm:
            svc = Pipeline([
                ('rbf', RBFSampler(gamma=best_params['rbf__gamma'], random_state=self.random_state)),
                ('sgd', SGDClassifier(
                    loss='hinge',
                    alpha=best_params['sgd__alpha'],
                    random_state=self.random_state
                ))
            ])
            svc.fit(X=train_X, y=train_y_numeric)
            return self.scaler, self.label_encoder, svc
        
        # Phase 2: Find optimal class weights for biased SVM
        best_class_weight = self._optimize_class_weights(
            train_X, train_y_numeric, best_params
        )
        
        # Train final model
        final_model = self._train_final_model(
            train_X, train_y_numeric, test_X, test_y_numeric, best_params, best_class_weight
        )
        
        return self.scaler, self.label_encoder, final_model
    
    def _optimize_hyperparameters(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray
    ) -> Dict[str, float]:
        """
        Optimize gamma and C using grid search.
        
        Args:
            train_X: Training features
            train_y: Training labels
            
        Returns:
            Best parameters dictionary
        """
        svc = Pipeline([
            ('rbf', RBFSampler(random_state=self.random_state)),
            ('sgd', SGDClassifier(random_state=self.random_state))
        ])
        
        parameters = {
            'sgd__loss': ('hinge',),
            'rbf__gamma': self.gamma_values,
            'sgd__alpha': self.alpha_values
        }
        
        scoring = self.biased_svm_score if self.use_biased_svm else 'f1'
        
        clf = GridSearchCV(
            svc,
            parameters,
            scoring=scoring,
            verbose=self.verbosity,
            n_jobs=self.n_jobs
        )
        
        clf.fit(X=train_X, y=train_y)
        
        return clf.best_params_
    
    def _optimize_class_weights(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        best_params: Dict[str, float]
    ) -> Dict[int, float]:
        """
        Find optimal class weights using cross-validation.
        
        Args:
            train_X: Training features
            train_y: Training labels
            best_params: Best gamma and C parameters
            
        Returns:
            Optimal class weight dictionary
        """
        # Convert to numpy if needed
        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.reset_index(drop=True).to_numpy()
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.reset_index(drop=True).to_numpy()
        
        print(f"train_X.shape: {train_X.shape}, train_y.shape: {train_y.shape}")
        
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        labels = [1, 0]  # Positive class first
        
        stats_df = []
        
        for i, class_weight in enumerate(self.class_weights):
            for j, (train_index, test_index) in enumerate(cv.split(train_X)):
                svc = Pipeline([
                    ('rbf', RBFSampler(gamma=best_params['rbf__gamma'], random_state=self.random_state)),
                    ('sgd', SGDClassifier(
                        loss='hinge',
                        alpha=best_params['sgd__alpha'],
                        class_weight=class_weight,
                        random_state=self.random_state
                    ))
                ])
                
                svc.fit(X=train_X[train_index], y=train_y[train_index])
                
                y_pred = svc.predict(train_X[test_index])
                confusion_matrix = sklearn.metrics.confusion_matrix(
                    train_y[test_index], y_pred, labels=labels
                )
                tn, fp, fn, tp = confusion_matrix.ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                stats_df.append({
                    'c_w_idx': i,
                    'specificity': specificity,
                    'sensitivity': sensitivity
                })
        
        # Find best class weight
        stats_df = pd.DataFrame(stats_df)
        stats_df = stats_df[stats_df['sensitivity'] != 0]
        stats_df = stats_df[['c_w_idx', 'specificity', 'sensitivity']]\
            .groupby('c_w_idx').mean().reset_index()
        stats_df['diff'] = (stats_df['sensitivity'] - self.sensitivity_target).abs()
        stats_df = stats_df.sort_values('diff')
        
        c_w_idx = int(stats_df.iloc[0]['c_w_idx'].item())
        
        print(f"C_w: {self.class_weights[c_w_idx]}, "
              f"specificity: {stats_df.iloc[0]['specificity']}, "
              f"sensitivity: {stats_df.iloc[0]['sensitivity']}")
        
        return self.class_weights[c_w_idx]
    
    def _train_final_model(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        best_params: Dict[str, float],
        class_weight: Dict[int, float]
    ) -> Pipeline:
        """
        Train final model and evaluate on test set.
        
        Args:
            train_X: Training features
            train_y: Training labels
            test_X: Test features
            test_y: Test labels
            best_params: Best hyperparameters
            class_weight: Optimal class weights
            
        Returns:
            Trained pipeline
        """
        svc = Pipeline([
            ('rbf', RBFSampler(gamma=best_params['rbf__gamma'], random_state=self.random_state)),
            ('sgd', SGDClassifier(
                loss='hinge',
                alpha=best_params['sgd__alpha'],
                class_weight=class_weight,
                random_state=self.random_state
            ))
        ])
        
        svc.fit(X=train_X, y=train_y)
        
        # Evaluate on test set
        y_pred = svc.predict(test_X)
        labels = [1, 0]
        confusion_matrix = sklearn.metrics.confusion_matrix(test_y, y_pred, labels=labels)
        tn, fp, fn, tp = confusion_matrix.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Test set... specificity: {specificity}, sensitivity: {sensitivity}")
        
        return svc