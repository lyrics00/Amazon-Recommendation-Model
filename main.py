from load_dataset import load_amazon_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
sample_data = load_amazon_dataset()

# Convert the sample data to a DataFrame
df = pd.DataFrame(sample_data)

# Ensure the DataFrame has the required columns
required_columns = ['user_id', 'parent_asin', 'rating', 'text', 'title', 'helpful_vote', 'verified_purchase']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

# Text preprocessing and feature extraction for text and title
vectorizer_text = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
X_text = vectorizer_text.fit_transform(df['text']).toarray()  # Convert to dense array

vectorizer_title = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
X_title = vectorizer_title.fit_transform(df['title']).toarray()  # Convert to dense array

# Convert helpful_vote and verified_purchase to numerical features
helpful_votes = df['helpful_vote'].values.reshape(-1, 1)
verified_purchase = df['verified_purchase'].astype(int).values.reshape(-1, 1)

# Combine all features for the training set
X = torch.cat((torch.FloatTensor(X_text), torch.FloatTensor(X_title), 
                torch.FloatTensor(helpful_votes), 
                torch.FloatTensor(verified_purchase)), dim=1)

y = torch.FloatTensor(df['rating'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

# Define the neural network model
class EnhancedRecommendationModel(nn.Module):
    def __init__(self, num_features):
        super(EnhancedRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 512)  # Increased size
        self.dropout1 = nn.Dropout(0.5)           # Dropout layer
        self.fc2 = nn.Linear(512, 256)            # Second hidden layer
        self.dropout2 = nn.Dropout(0.5)           # Dropout layer
        self.fc3 = nn.Linear(256, 128)            # Third hidden layer
        self.fc4 = nn.Linear(128, 1)              # Output layer

    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x) * 5  # Scale output to [0, 5]

# Create the model
num_features = X.shape[1]  # Update to include all features
model = EnhancedRecommendationModel(num_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Lists to store loss and RMSE values for plotting
train_losses = []
val_rmses = []

# Training the model
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(predictions.view(-1), y_train_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_rmse = torch.sqrt(criterion(val_predictions.view(-1), y_test_tensor))
    
    # Store loss and RMSE for plotting
    train_losses.append(loss.item())
    val_rmses.append(val_rmse.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val RMSE: {val_rmse.item():.4f}')

# Plotting the training loss and validation RMSE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_rmses, label='Validation RMSE', color='orange')
plt.title('Validation RMSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test data
model.eval()
with torch.no_grAmad():
    test_predictions = model(X_test_tensor)
    test_predictions = torch.clamp(test_predictions, 0, 5)  # Clamp predictions to [0, 5]

# Calculate RMSE
test_rmse = torch.sqrt(criterion(test_predictions.view(-1), y_test_tensor))
print(f'Test RMSE: {test_rmse.item():.4f}')

