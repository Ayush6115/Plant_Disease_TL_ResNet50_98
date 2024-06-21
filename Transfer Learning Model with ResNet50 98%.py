#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# # Constants

# In[2]:


img_size = 256
batch_size = 32
epochs = 5


# # Loading dataset

# In[3]:


dataset_path = r'D:\Minor-Sem-4\PlantVillage_dataset'

full_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    seed=42
)


# # Splitting the dataset into train, validation, and test sets

# In[4]:


train_ds_size = int(0.6 * len(full_ds))
val_ds_size = int(0.2 * len(full_ds))
test_ds_size = len(full_ds) - train_ds_size - val_ds_size

train_ds = full_ds.take(train_ds_size)
val_ds = full_ds.skip(train_ds_size).take(val_ds_size)
test_ds = full_ds.skip(train_ds_size + val_ds_size).take(test_ds_size)


# # Sizes of train, validation, and test datasets

# In[5]:


train_ds_size = len(train_ds)
val_ds_size = len(val_ds)
test_ds_size = len(test_ds)

print("Training dataset size:", train_ds_size)
print("Validation dataset size:", val_ds_size)
print("Test dataset size:", test_ds_size)


# # Data augmentation

# In[6]:


data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.1),
])


# # Loading pre-trained ResNet50

# In[7]:


base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
output_layer = tf.keras.layers.Dense(4, activation='softmax')


# # Creating the model

# In[8]:


inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = output_layer(x)
model = tf.keras.Model(inputs, outputs)


# # Compiling the model

# In[9]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# # Training the model

# In[10]:


history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs)


# # Evaluating the model

# In[11]:


test_loss, test_acc = model.evaluate(val_ds)
print(f"Test Accuracy: {test_acc}")


# In[21]:


import visualkeras
visualkeras.layered_view(model, legend=True, scale_xy=0.8)


# In[22]:


model.summary()


# In[14]:


from sklearn.metrics import classification_report

# Convert validation dataset to numpy arrays
val_images = []
val_labels = []
for images, labels in val_ds:
    val_images.append(images.numpy())
    val_labels.append(labels.numpy())

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

# Predict labels for validation dataset
val_predictions = model.predict(val_images)
val_pred_labels = np.argmax(val_predictions, axis=-1)

# Generate classification report
class_names = full_ds.class_names
report = classification_report(val_labels, val_pred_labels, target_names=class_names)

print(report)


# # Plotting accuracy and loss curve

# In[15]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# # Iterate through the test dataset and make predictions

# In[16]:


num_images_to_display = 6
num_cols = 3
num_rows = (num_images_to_display + 1)
plt.figure(figsize=(12, 14))

for idx, (images_batch, labels_batch) in enumerate(test_ds.take(1)):
    predicted_labels = model.predict(images_batch)
    predicted_class_indices = np.argmax(predicted_labels, axis=1)
    predicted_class_names = [class_names[idx] for idx in predicted_class_indices]

    # Shuffle the indices for displaying
    indices = np.arange(len(images_batch))
    np.random.shuffle(indices)
    shuffled_indices = indices[:num_images_to_display]

    # Display images and predictions
    for i, idx in enumerate(shuffled_indices):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images_batch[idx].numpy().astype("uint8"))
        plt.title(f"Predicted: {predicted_class_names[idx]}\nActual: {class_names[labels_batch[idx]]}")
        plt.axis("off")

plt.tight_layout()
plt.show()


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Iterate through the test dataset and make predictions
for images_batch, labels_batch in test_ds:
    predicted_labels_batch = model.predict(images_batch)
    predicted_class_indices = np.argmax(predicted_labels_batch, axis=1)
    
    true_labels.extend(labels_batch.numpy())
    predicted_labels.extend(predicted_class_indices)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # Printing no. of correct and wrong predictions

# In[18]:


# Number of correct predictions
correct_predictions = np.sum(np.array(true_labels) == np.array(predicted_labels))

# Number of wrong predictions
total_predictions = len(true_labels)
wrong_predictions = total_predictions - correct_predictions

print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of wrong predictions: {wrong_predictions}")


# # Save the model

# In[19]:


model.save('resnet_model_98%.h5')

