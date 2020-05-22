import tensorflow as tf

class Trainer():
    
    def __init__(self, model, optimizer=tf.optimizers.RMSprop()):
        self.model = model
        self.optimizer = optimizer
    
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