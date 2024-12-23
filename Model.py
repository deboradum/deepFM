from tinygrad import Tensor, nn


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


# TODO: Experiment with different initializations
# TODO: after that, a dense real-value feature vector is gener-
#       ated, which is finally fed into the sigmoid function for CTR
#       prediction: yDN N = σ(W |H|+1 · aH + b|H|+1), where |H|
#       is the number of hidden layers.
# https://arxiv.org/abs/1703.04247
class DeepNet:
    def __init__(self, input_dim, num_layers, hidden_dim):
        self.l1 = Sequential(
            nn.Linear(input_dim, hidden_dim),  # TODO: Try out layer norm as well.
            nn.BatchNorm(hidden_dim),
        )

        self.layers = [
            Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm(hidden_dim),  # TODO: Try out layer norm as well.
            )
            for _ in range(num_layers)
        ]
        self.final_layer = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        x = self.l1(x).dropout(0.5).relu()
        for layer in self.layers:
            x = layer(x).dropout(0.5).relu()

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
        dense_x_o2_list = [
            embed_layer(
                x[
                    :, sum(self.feature_sizes[:i]) : sum(self.feature_sizes[: i + 1])
                ].argmax(axis=1)
            )
            for i, embed_layer in enumerate(self.o2_embeddings)
        ]
        dense_x_o2 = Tensor.cat(*dense_x_o2_list, dim=1)

        y_fm = self.fm(x, dense_x_o2_list)
        y_dnn = self.deep(dense_x_o2).sum(axis=1)

        return y_fm + y_dnn

    # https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
    def fm(self, x, dense_x_o2_list):
        # First order part of FM
        # \sum^{n}_{i=1} w_{i} x_{i}
        dense_x_o1_list = [
            embed_layer(
                x[
                    :, sum(self.feature_sizes[:i]) : sum(self.feature_sizes[: i + 1])
                ].argmax(axis=1)
            )
            for i, embed_layer in enumerate(self.o1_embeddings)
        ]
        fm_o1 = Tensor.cat(*dense_x_o1_list, dim=1).sum(axis=1)

        # Second order part of FM
        # 0.5 * \sum^{k}_{f=1}( (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2} - \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i} )
        # t1 = (\sum^{n}_{i=1} v_{i, f} x_{i} )^{2}
        # t2 = \sum^{n}_{i=1} v^{2}_{i, f} x^{2}_{i}
        t1 = sum(dense_x_o2_list).square()
        t2 = sum([e * e for e in dense_x_o2_list])
        fm_o2 = 0.5 * (t1 - t2).sum(axis=1)

        return self.bias + fm_o1 + fm_o2
