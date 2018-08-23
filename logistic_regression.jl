function sigmoid(S)
    return 1./(1 + exp.(-S))
end

prob(w, X) = sigmoid(X*w)

function loss(w, X, y, lamda)
    z = prob(w, X)
    return -mean(y.*log.(z) + (1-y).*log.(1-z)) + 0.5*lamda/size(X, 2)*sum(w.*w)
end

function logistic_regression(w_init, X, y, lam = 0.001, lr = 0.1, nepoches = 2000)
    N, d = size(X)
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)] # store history of loss in loss_hist
    ep = 0
    while ep < nepoches
        ep += 1
        mix_ids = randperm(N)
        for i in mix_ids
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi*w)
            w = w - lr.*((zi - yi).*xi + lam.*w)
        end
        append!(loss_hist,loss(w, X, y, lam))
        if norm(w - w_old)/d < 1e-6
            break
        end
        w_old = w
    end
    return w, loss_hist
end

X = [0.50 0.75 1.00 1.25 1.50 1.75 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 4.00 4.25 4.50 4.75 5.00 5.50]'
y = [0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1]'
# bias trick
size(X)
size(y)
Xbar = hcat(X, ones(size(y, 1), 1))
w_init = randn(size(Xbar, 2), 1)
lam = 0.0001
size(w_init)
w, loss_hist = logistic_regression(w_init, Xbar, y, lam, 0.05, 500)
println(w)
println(loss(w, Xbar, y, lam))

# Plot results
using Plots
plot(loss_hist)
