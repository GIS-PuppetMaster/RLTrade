import numpy as np
import torch
from Tianshou.Net.NBeats import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, stocks, input_dim, output_dim = 32, 60, 180, 32, 2

    # Definition of the model.
    model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=2,
                      thetas_dims=(4, 4, 4), share_weights_in_stack=True, hidden_layer_units=64,device=device)

    # Definition of the objective function and the optimizer.
    # model.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, time_steps, stocks, input_dim)).astype(np.float)
    y = torch.tensor(np.mean(x, axis=1, keepdims=True)).cuda()

    # Split data into training and testing datasets.
    c = 0
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.L1Loss()
    for i in range(20):
        res = model(torch.tensor(x_train,device=device, dtype=torch.float))[1]
        l = loss(res, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == '__main__':
    main()