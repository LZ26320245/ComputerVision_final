import tensorflow as tf
import matplotlib.pyplot as plt

image_size = (224,224)
batch_size = 16
epochs = 10

tdata = tf.keras.preprocessing.image_dataset_from_directory(
    './training/',
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

testdata = tf.keras.preprocessing.image_dataset_from_directory(
    './testing/',
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = tdata.class_names
num_classes = len(class_names)
print("Classes:", class_names)

tdata = tdata.prefetch(buffer_size=batch_size)
testdata = testdata.prefetch(buffer_size=batch_size)


input_shape = (image_size[0], image_size[1], 3)

feature_model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

feature_model.trainable = False  # freeze

model = tf.keras.models.Sequential([
    feature_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    tdata,
    epochs=epochs,
    validation_data=testdata
)

model.save("model_VGG16.keras")

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig("acc.png")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig("loss.png")