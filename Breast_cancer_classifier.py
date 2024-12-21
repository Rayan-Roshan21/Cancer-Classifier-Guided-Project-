from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt

# Load the breast cancer dataset from scikit-learn
breast_cancer_data = load_breast_cancer()

# Split the dataset into training and validation sets (80% train, 20% validate)
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, 
    breast_cancer_data.target, 
    test_size=0.2, 
    random_state=10
)

# Initialize lists to store K values and their corresponding accuracies
k_list = []
accuracies = []

# Test KNN classifier performance for K values from 1 to 100
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)  # Create a KNN classifier with k neighbors
    classifier.fit(training_data, training_labels)    # Train the classifier
    k_list.append(k)                                  # Add the current K value to the list
    accuracies.append(classifier.score(validation_data, validation_labels))  # Store the validation accuracy

# Plot the relationship between K values and validation accuracy
plt.plot(k_list, accuracies)
plt.title("Breast Cancer Classifier Accuracy")  # Title of the plot
plt.xlabel("K Values")                          # X-axis label
plt.ylabel("Validation Accuracy")               # Y-axis label
plt.show()                                      # Display the plot
