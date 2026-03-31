
import tensorflow as tf
from tensorflow.keras import layers

# Function: create_cnn_model - Helper routine for create cnn model logic.
# Parameters: none.
def create_cnn_model():
    INPUT_SHAPE = (224, 224, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Fine-tune later if needed
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    m = create_cnn_model()
    m.summary()
