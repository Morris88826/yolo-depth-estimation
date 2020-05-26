import tensorflow as tf


class Trainer():
    
    def __init__(self, model, lr=1e-5, decay=0.9):
        self.model = model
        self.optimizer = tf.optimizers.RMSprop(learning_rate=lr, decay=0.01)
        self.lr = lr
    
    def train(self, X, y):
        with tf.GradientTape() as tape:
            _, y_predict = self.model(X)
            loss_value = self.MSE_Loss(y, y_predict)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


        return loss_value.numpy().mean()


    def MSE_Loss(self, target_y, predicted_y):
        # Use mean square error
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    def depth_loss_function(self, y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
        # y_true = tf.convert_to_tensor(y_true)
        # Point-wise depth
        l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) + tf.keras.backend.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = tf.keras.backend.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta

        return (w1 * l_ssim) + (w2 * tf.keras.backend.mean(l_edges)) + (w3 * tf.keras.backend.mean(l_depth))