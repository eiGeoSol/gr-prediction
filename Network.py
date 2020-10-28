def network(x_train, y_train, x_val, y_val, x_test, y_test):
    """

    This code execute the network and give as output the trained model
    
    x_train : input train samples 
    y_train : output train samples 
    x_val : input validation samples 
    y_val : output validation samples 
    x_test : input test samples 
    y_test : output test samples 

    """
    
    from keras import layers, models, optimizers, callbacks
    
    # Network parameters
    epochs = 200
    batch_size = 32
    lr = 4e-3
    optimizer = optimizers.Adam(lr=lr)
    loss = 'mse'
    metrics = 'mean_absolute_error'

    # Network
    inputs = []
    conv_1 = []
    pool_1 = []
    for n in range(len(x_train)):
        inputs.append(models.Input(shape=x_train[n].shape[1:]))
        conv_1.append(layers.Conv2D(128, (5, 5), padding='same', activation='relu')(inputs[n]))
        pool_1.append(layers.MaxPooling2D((2, 2))(conv_1[n]))

    if len(x_train) != 1:
        cont = layers.add([pool_1[n] for n in range(len(x_train))])
        conv_2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(cont)

    else:
        conv_2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(pool_1[0])

    pool_2 = layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(pool)
    pool_3 = layers.MaxPooling2D((2, 2))(conv_3)

    flat = layers.Flatten()(pool_3)

    dense1 = layers.Dense(80)(flat)
    norm1 = layers.BatchNormalization()(dense1)
    act1 = layers.Activation('elu')(norm1)

    dense2 = layers.Dense(32)(act1)
    norm2 = layers.BatchNormalization()(dense2)
    act2 = layers.Activation('elu')(norm2)

    dense3 = layers.Dense(16)(act2)
    norm3 = layers.BatchNormalization()(dense3)
    act3 = layers.Activation('elu')(norm3)

    output = layers.Dense(len(y_train[0]))(act3)

    model = models.Model(inputs=[inputs[n] for n in range(len(x_train))], outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.summary()


    # Function that reduce a learning rate during a run
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.25, patience=25, min_lr=5e-8)

    history = model.fit([x_train[n] for n in range(len(x_train))], y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[reduce_lr],
                        validation_data=([x_val[n] for n in range(len(x_train))], y_val)
                        )

    score = model.evaluate([x_test[n] for n in range(len(x_train))], y_test, verbose=0)
    print("Error: %f" % (score[1]))
    
    return model

