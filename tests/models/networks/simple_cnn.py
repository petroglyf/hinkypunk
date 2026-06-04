import flax.linen as nn


class SimpleClassifier(nn.Module):
    c_hid: int
    num_classes: int
    act_fn: str
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool=True, **kwargs):
        act_fn = getattr(nn.activation, self.act_fn)
        while x.shape[1] > 4:
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # pyrefly: ignore[missing-argument]  # 32x32 => 16x16
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)  # pyrefly: ignore[missing-argument]
            x = act_fn(x)
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)  # pyrefly: ignore[missing-argument]
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)  # pyrefly: ignore[missing-argument]
            x = act_fn(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.num_classes)(x)  # pyrefly: ignore[missing-argument]
        x = nn.log_softmax(x)
        return x
