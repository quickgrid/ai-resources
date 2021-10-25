import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7])
Y = np.array([2, 3, 4, 5, 6, 7, 8])
w_gradient = 0
b_gradient = 0
w, b = 0.5, 0.5

learning_rate = .01
loss = 0
EPOCHS = 2000
N = len(Y)


for i in range(EPOCHS):

    w_gradient = 0
    b_gradient = 0
    loss = 0

    for j in range(N):

        # Predict
        Y_pred = (w * X[j]) + b

        # Loss
        loss += np.square(Y_pred - Y[j]) / (2.0 * N)

        # Backprop
        grad_y_pred = (2 / N) * (Y_pred - Y[j])
        w_gradient += (grad_y_pred * X[j])
        b_gradient += (grad_y_pred)

    # Optimize
    w -= (w_gradient * learning_rate)
    b -= (b_gradient * learning_rate)

    # Print loss
    if i % 100 == 0:
        print(loss)


print("\n\n")
print("LEARNED:")
print(w, b)
print("\n")
print("TEST:")
print(np.round(b + w * (-2)))
print(np.round(b + w * 0))
print(np.round(b + w * 1))
print(np.round(b + w * 6))
print(np.round(b + w * 3000))

# Expected: 30001, but gives 30002.
# Training for 3000 epochs will give expected result.
# For simple demo with less training data and small input range 2000 in enough
print(np.round(b + w * 30000))

