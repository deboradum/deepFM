from tinygrad import Tensor, nn


# TODO: Experiment with different initializations
# TODO: after that, a dense real-value feature vector is gener-
#       ated, which is finally fed into the sigmoid function for CTR
#       prediction: yDN N = σ(W |H|+1 · aH + b|H|+1), where |H|
#       is the number of hidden layers.
# TODO: Add dropout + batch/ layer norm
# https://arxiv.org/abs/1703.04247
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

        self.o1_embeddings = [
            nn.Embedding(field_size, 1) for field_size in self.feature_sizes
        ]
        self.o2_embeddings = [
            nn.Embedding(field_size, k) for field_size in self.feature_sizes
        ]

        self.bias = Tensor.rand(1)

        self.deep = DeepNet(
            input_dim=self.m * k,
            num_layers=6,
            hidden_dim=1024,
        )

    def __call__(self, x):
        # convert sparse x to dense x by concatenating feature embeddings
        dense_x_o1_list = [
            embed_layer(
                x[
                    :, sum(self.feature_sizes[:i]) : sum(self.feature_sizes[: i + 1])
                ].argmax(axis=1)
            )
            for i, embed_layer in enumerate(self.o1_embeddings)
        ]
        dense_x_o1 = Tensor.cat(*dense_x_o1_list, dim=1)

        dense_x_o2_list = [
            embed_layer(
                x[
                    :, sum(self.feature_sizes[:i]) : sum(self.feature_sizes[: i + 1])
                ].argmax(axis=1)
            )
            for i, embed_layer in enumerate(self.o2_embeddings)
        ]
        dense_x_o2 = Tensor.cat(*dense_x_o2_list, dim=1)

        y_fm = self.fm(dense_x_o1, dense_x_o2_list, dense_x_o2)
        y_dnn = self.deep(dense_x_o2).sum(axis=1)

        return y_fm + y_dnn + self.bias

    # From # https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf:
    # 0.5 * \sum^{k}_{f=1}( (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2} - \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i} )
    # t1 = (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2}
    # t2 = \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i}
    def fm(self, fm_o1, dense_x_o2_list, dense_x_o2):
        # equivalent to (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2}, because x is one hot encoded
        t1 = sum(dense_x_o2_list).square()
        t2 = sum([e*e for e in dense_x_o2_list])

        fm_o2 = 0.5 * (t1 - t2).sum(axis=1)

        return fm_o1.sum(axis=1) + fm_o2.sum(axis=1)


d = DeepFM([3, 5, 2], 2)
test = Tensor([[0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 1, 0, 1, 0]])
print(test.numpy(), test.shape)
d(test)