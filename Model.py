import tinygrad
from tinygrad import Tensor, nn


class FactorizationMachine:
    def __init__(self):
        return

    def __call__(self, x):
        return


# TODO: Experiment with different initializations
class DeepNet:
    def __init__(self, input_dim, num_layers, hidden_dim):
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim)]
        self.final_layer = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        x = self.l1(x).relu()
        for layer in self.layers:
            x = layer(x).relu()

        return self.final_layer(x)


class DeepFM:
    # m: Number of feature fields
    # k: embedding size per feature field
    def __init__(self, feature_sizes, k):
        self.m = len(feature_sizes)
        self.feature_sizes = feature_sizes

        self.o1_embeddings = [nn.Embedding(field_size, 1) for field_size in self.feature_sizes]
        self.o2_embeddings = [nn.Embedding(field_size, k) for field_size in self.feature_sizes]

        self.fm = FactorizationMachine()
        self.deep = DeepNet(
            input_dim=self.m * k,
            num_layers=6,
            hidden_dim=1024,
        )

    def __call__(self, x):
        # convert sparse x to dense x by concatenating feature embeddings
        prev_i = 0

        dense_x_o2 = [
            embed_layer(
                x[
                    :, sum(self.feature_sizes[:i]) : sum(self.feature_sizes[: i + 1])
                ].argmax(axis=1)
            )
            for i, embed_layer in enumerate(self.o2_embeddings)
        ]
        dense_x_o2 = Tensor.cat(*dense_x_o2, dim=1)

        y_fm = self.fm(dense_x_o2)
        y_dnn = self.deep(dense_x_o2)

        return (y_fm + y_dnn).sigmoid()


d = DeepFM([3, 5, 2], 2)
test = Tensor([[0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 1, 0, 1, 0]])
print(test.numpy(), test.shape)
d(test)