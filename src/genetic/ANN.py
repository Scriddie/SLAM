import numpy as np
import pickle
np.set_printoptions(suppress=True)

class ANN:

    def __init__(self, input_dims, output_dims, hidden_dims,
    eta=0.15, reg=0):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.weight_matrices = []
        self.eta = eta
        self.reg = reg
        self.create_architecture()
        self.prev_activations = []

    def create_architecture(self):
        prev_dim = self.input_dims
        for hidden_dim in self.hidden_dims:
            self.weight_matrices.append(
                (np.random.rand(hidden_dim, prev_dim+1)-0.5)/1)
            prev_dim = hidden_dim
            
        self.weight_matrices.append(
            (np.random.rand(self.output_dims, prev_dim+1)-0.5)/1)

    def show(self):
        for i, l in enumerate(self.weight_matrices):
            print(f"Weights {i} with shape {l.shape}")
            print(f"{np.round(l, 2)}\n")

    def predict(self, x, feedback=True):
        """
            @param x: observation to predict on
            @param feedback: feedback of prev last hidden layer
        """
        batch_size = x.shape[1]  # one column is one observation
        x = np.concatenate((np.ones((1, batch_size)), x), axis=0)
        activations = [x]
        for i, theta in enumerate(self.weight_matrices):

            # TODO: check this once more
            if feedback:
                # for our last layer, use our prev activations
                if i == (len(self.weight_matrices)-2):
                    num_obs = max(1, batch_size)
                    if len(self.prev_activations) > 0:
                        last_hidden = self.prev_activations[-2]
                    else:  # initialize with random activations
                        missing = theta.shape[1] - x.shape[0]
                        last_hidden = np.random.uniform(0, 1, size=missing).reshape(-1, 1)
                        last_hidden = np.repeat(last_hidden, num_obs, axis=1)

                    x = np.concatenate((x, last_hidden), axis=0)

            z = np.dot(theta, x).reshape(-1, max(1, batch_size))
            z = self.sigmoid(z, deriv=False)
            x = np.concatenate((np.ones((1, batch_size)), z), axis=0)
            activations.append(x)
        
        self.prev_activations = [i[1:, :] for i in activations]
        return self.prev_activations[-1]

    def backpropagate(self, x, y):
        # TODO: wont work with feedback just yet
        batch_size = x.shape[1]
        activations = self.predict(x)
        error_deriv = np.subtract(activations[-1], y)
        delta = []
        for i in range(1, len(self.weight_matrices)+1):
            # print(f"step {i} of backpropagation")
            if len(delta) == 0:
                delta = np.multiply(
                    error_deriv,
                    self.sigmoid(activations[-i], deriv=True)
                )
            else:
                delta = np.multiply(
                    np.dot(
                        np.transpose(self.weight_matrices[-(i-1)]),
                        delta
                    ),
                    self.sigmoid(activations[-i], deriv=True)
                )[1:, :]
            # delta_shape = delta.shape
            # print(f"delta shape: {delta_shape}")
            gradient = np.dot(
                delta,
                np.transpose(activations[-(i+1)])
            )

            # # Smth like L1 regularization
            gradient += np.multiply(
                np.sign(self.weight_matrices[-i]),
                np.where(self.weight_matrices[-i] != 0, self.reg, 0)
            )

            # grad_shape = gradient.shape
            # print(f"gradient shape: {grad_shape}")
            self.weight_matrices[-i] -= self.eta * (1/batch_size) * gradient
        
    def sigmoid(self, x, deriv=False):
        sig = 1/(1 + np.exp(-x))
        if deriv:
            return np.multiply(sig, 1-sig)
        else:
            return sig

    @staticmethod
    def train_backprop(cn, x, y):

        # initial prediction
        result = cn.predict(x)
        print("\nPredictions before training:")
        for i in result:
            print(np.round(i, 3))
            print("\n")

        # train
        for i in range(30000):
            cn.backpropagate(x, y)

        # predict again
        print("\nPredictions after training:")
        result = cn.predict(x)
        for i in result:
            print(np.round(i, 3))
            print("\n")

        mse = np.mean(np.square(np.subtract(result[-1], y)))
        print(f"\n\nMSE: {mse}")

    def train_on_function(self, cn, x, func):
        y = func(x)
        pass 
    
    def save(self, file_path):
        pickle.dump(self, open(file_path, "wb"))
        
    @staticmethod
    def load(file_path):
        return pickle.load(open(file_path, "rb"))
    

if __name__ == "__main__":
    cn = ANN(input_dims=8, output_dims=8, hidden_dims=[3])
    cn.show()
    x = np.eye(8)
    y = x
    cn.train_backprop(cn, x, y)

