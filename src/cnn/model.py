from keras.models import Model
from keras.layers import (
    Dense,
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    LeakyReLU,
    Concatenate,
)


# multi class classification
def create_model(
    img_size: tuple[int, int, int], num_classes: int, num_extra_features: int
) -> Model:
    height, width, num_color_channels = img_size

    input_image = Input(
        shape=(
            height,
            width,
            num_color_channels,
        )
    )
    extra_features = Input(shape=(num_extra_features,))

    x = Conv2D(1, (3, 3), padding="same", activation="relu")(input_image)
    x = MaxPooling2D((5, 5))(x)

    x = Conv2D(2, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((5, 5))(x)

    post_convolution = Flatten()(x)
    x = Concatenate()([post_convolution, extra_features])

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    out_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[input_image, extra_features], outputs=out_layer)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model
