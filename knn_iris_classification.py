import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# ------------ Utility: Write output to file --------------
def write_to_file(text):
    with open("results.txt", "a") as f:
        f.write(text + "\n")



# ------------ MAIN KNN Function -------------------------
def train_and_evaluate_knn():

    write_to_file("===== KNN Classification Results =====\n")

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Try different K values
    k_values = list(range(1, 16, 2))
    accuracies = []

    print("\nComparing K values:")
    write_to_file("Comparing K values:")

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"K={k} -> Accuracy={acc:.4f}")
        write_to_file(f"K={k} -> Accuracy={acc:.4f}")

    # 5. Choose best K
    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies)

    print(f"\nBest K = {best_k}, Best Accuracy = {best_acc:.4f}")
    write_to_file(f"\nBest K = {best_k}, Best Accuracy = {best_acc:.4f}")

    # Plot Accuracy vs K
    plt.figure()
    plt.plot(k_values, accuracies, marker="o")
    plt.title("Accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    # 6. Train final model
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_pred_best = best_knn.predict(X_test_scaled)

    # Evaluation
    acc_final = accuracy_score(y_test, y_pred_best)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best, target_names=class_names)

    print("\nFinal Accuracy:", acc_final)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    write_to_file("\nFinal Accuracy: " + str(acc_final))
    write_to_file("\nConfusion Matrix:\n" + str(cm))
    write_to_file("\nClassification Report:\n" + str(report))

    # Plot Confusion Matrix
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color="black")

    plt.tight_layout()
    plt.show()

    # 7. Decision boundary
    visualize_decision_boundary(iris, feature_indices=(0, 2), k=best_k)



# -------------------- DECISION BOUNDARY ------------------
def visualize_decision_boundary(iris, feature_indices=(0, 2), k=5):
    X = iris.data[:, feature_indices]
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot points
    for idx, cls in enumerate(class_names):
        plt.scatter(
            X_train_scaled[y_train == idx, 0],
            X_train_scaled[y_train == idx, 1],
            edgecolor='k',
            label=cls
        )

    plt.title("Decision Boundary")
    plt.legend()
    plt.show()



# ======================== RUN HERE ========================
if __name__ == "__main__":
    open("results.txt", "w").close()      # Clear old results
    train_and_evaluate_knn()
