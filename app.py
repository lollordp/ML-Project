from shiny import ui, render, App
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import numpy as np
from ucimlrepo import fetch_ucirepo 


# Load data
#df = pd.read_csv(r'/Users/stephaneleboyer/Desktop/UNIVERSITE/MASTER MTF/ML in Finance/CourseWork/1/drug200.csv')
#X = df.iloc[:, :5]
#Y = df.iloc[:, -1]

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
Y = breast_cancer_wisconsin_diagnostic.data.targets 
df=X
df['Diagnosis']=Y
X = df.iloc[:, :30]
Y = df.iloc[:, -1]
Y = Y.map({'M': 1, 'B': 0})

# Encode categorical variables
#Y = Y.map({'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4, 'drugY': 5})
#X['Sex'] = X['Sex'].map({'F': 1, 'M': 0})
#X['BP'] = X['BP'].map({'HIGH': 1, 'LOW': 0, 'NORMAL': 2})
#X['Cholesterol'] = X['Cholesterol'].map({'HIGH': 1, 'NORMAL': 2})

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=12)

# Define UI
app_ui = ui.page_fluid(
    # Step 1: Choose model type
    ui.input_select("model_type", "Select Model Type", choices=["Decision Tree", "Random Forest"]),
    
    # Step 2: Choose what to display
    ui.output_ui("display_selector"),
    
    # Step 3: Choose Gini or Entropy (only for tree-based models)
    ui.output_ui("criterion_selector"),
    
    # Output plot
    ui.output_plot("plot_output")
)

# Define Server
def server(input, output, session):
    
    @output
    @render.ui
    def display_selector():
        """Dynamically update options based on model type"""
        options = ["Tree", "Confusion Matrix", "ROC Curve", "Feature Importance"]
        if input.model_type() == "Random Forest":
            options.append("Error vs. Number of Trees")  # New option for RF
        return ui.input_select("display_option", "Select What to Show", choices=options)
    
    @output
    @render.ui
    def criterion_selector():
        """Show criterion selector only for tree-based models"""
        return ui.input_select("var", "Select Criterion", choices=["gini", "entropy"])
    
    @output
    @render.plot
    def plot_output():
        """Render output based on user selections"""
        model_type = input.model_type()
        display_option = input.display_option()
        criterion = input.var()

        # Train the selected model
        if model_type == "Decision Tree":
            model = DecisionTreeClassifier(ccp_alpha=0.01, random_state=67, criterion=criterion)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=67, criterion=criterion)
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # Plot Decision Tree or an example Random Forest tree
        if display_option == "Tree":
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            if model_type == "Decision Tree":
                tree.plot_tree(model, feature_names=X_train.columns, class_names=['Benign', 'Malignant'], filled=True, ax=ax)
                ax.set_title("Decision Tree Visualization")
            else:
                # Pick one tree from the Random Forest (e.g., the first estimator)
                tree.plot_tree(model.estimators_[0], feature_names=X_train.columns, class_names=['Benign', 'Malignant'], filled=True, ax=ax)
                ax.set_title("Example Tree from Random Forest (One of Many)")
            return fig

        # Plot Confusion Matrix
        elif display_option == "Confusion Matrix":
            cm = confusion_matrix(Y_test, Y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix ({model_type})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            return fig
        
        # Plot ROC Curve
        elif display_option == "ROC Curve":
            # Get class probabilities
            probs = model.predict_proba(X_test)  # Works for both Decision Tree and Random Forest

            # Set up the plot
            fig, ax = plt.subplots(figsize=(6, 4))

            # Compute ROC curve for each class
            for i, class_label in enumerate(model.classes_):  
                fpr, tpr, _ = roc_curve((Y_test == class_label).astype(int), probs[:, i])  
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {class_label} (AUC: {auc_score:.2f})")

            # Diagonal reference line
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title(f"ROC Curve ({model_type})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            
            return fig

        # Plot Feature Importance
        elif display_option == "Feature Importance":
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(X_train.columns, importances, color="teal")
            ax.set_title(f"Feature Importance ({model_type})")
            ax.set_xlabel("Importance")
            return fig

        # NEW: Plot Error vs. Number of Trees (Only for Random Forest)
        elif display_option == "Error vs. Number of Trees" and model_type == "Random Forest":
            num_trees = list(range(1, 101, 10))  # Test 1, 10, 20, ..., 100 trees
            errors = []

            for n in num_trees:
                rf = RandomForestClassifier(n_estimators=n, random_state=67, criterion=criterion)
                rf.fit(X_train, Y_train)
                y_pred_rf = rf.predict(X_test)
                error = 1 - accuracy_score(Y_test, y_pred_rf)  # Error = 1 - Accuracy
                errors.append(error)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(num_trees, errors, marker="o", linestyle="-", color="red")
            ax.set_title("Error vs. Number of Trees in Random Forest")
            ax.set_xlabel("Number of Trees")
            ax.set_ylabel("Error (1 - Accuracy)")
            ax.grid(True)

            return fig

# Run App
app = App(app_ui, server)


