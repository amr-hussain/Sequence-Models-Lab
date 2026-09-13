"""
Vanilla RNN, implemented in pure NumPy: forward pass AND manual backprop
through time (BPTT). No autograd anywhere in this file.

This is the single most important file in the lab. If you only deeply
understand one thing, make it this one — everything LSTM/GRU/Transformer do
is a reaction to what goes wrong here.

Recurrence (what you're implementing):
    a_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    h_t = a_t                                          (RNN has no separate output nonlinearity here)
    y_t = W_hy @ h_t + b_y
    p_t = softmax(y_t)

Shapes (all NumPy arrays):
    x_t   : (input_size,)
    h_t   : (hidden_size,)
    W_xh  : (hidden_size, input_size)
    W_hh  : (hidden_size, hidden_size)
    b_h   : (hidden_size,)
    W_hy  : (output_size, hidden_size)
    b_y   : (output_size,)

Loss: cross-entropy on the final output only (or summed over all timesteps —
see the `mode` argument), matching the copy-task setup where only the tail
of the sequence carries labeled targets.
"""
import numpy as np


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed) #default_rng is for consistent ranges
        scale = 0.01
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = rng.standard_normal((hidden_size, input_size)) * scale
        self.W_hh = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.b_h = np.zeros(hidden_size)
        self.W_hy = rng.standard_normal((output_size, hidden_size)) * scale
        self.b_y = np.zeros(output_size)

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------
    def forward(self, xs, h0=None):
        """
        Args:
            xs: list/array of one-hot vectors, length T, each shape (input_size,)
            h0: initial hidden state, shape (hidden_size,), defaults to zeros

        Returns:
            ys:  list of length T, each (output_size,)   -- pre-softmax logits
            ps:  list of length T, each (output_size,)   -- softmax probabilities
            hs:  list of length T+1, each (hidden_size,) -- hs[0] = h0, hs[t+1] = h_t
                 (keep ALL hidden states — you need every one of them for BPTT)

        TODO(1): Implement the forward recurrence.
        For t in range(T):
            1. compute a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h
            2. h_t = tanh(a_t)
            3. y_t = W_hy @ h_t + b_y
            4. p_t = softmax(y_t)
            5. store h_t into hs, y_t into ys, p_t into ps

        Hint: store hs[0] = h0 first, then append. You need hs[t] AND hs[t-1]
        available during the backward pass, for every t.
        """
        T = len(xs)
        h0 = np.zeros(self.hidden_size) if h0 is None else h0
        hs = [h0]
        ys = []
        ps = []
        for t in range(T):
            a_t = self.W_xh @ xs[t] + self.W_hh @ hs[t] + self.b_h
            h_t = np.tanh(a_t)
            y_t = self.W_hy @ h_t + self.b_y
            p_t = softmax(y_t)

            hs.append(h_t)
            ys.append(y_t)
            ps.append(p_t)

        return ys, ps, hs

    # ------------------------------------------------------------------
    # BACKWARD PASS (manual BPTT) — the part that teaches you everything
    # ------------------------------------------------------------------
    def backward(self, xs, ys_target, ps, hs):
        """
        Args:
            xs:        list of one-hot input vectors, length T
            ys_target: list of integer class labels, length T (target token index at each t,
                       or None at positions that don't contribute to the loss —
                       skip those positions entirely when accumulating gradients)
            ps:        softmax probabilities from forward(), length T
            hs:        hidden states from forward(), length T+1

        Returns:
            grads: dict with keys 'W_xh','W_hh','b_h','W_hy','b_y', each the
                   gradient of the total loss w.r.t. that parameter (same shape
                   as the parameter itself)
            grad_norms_per_t: list of length T, the L2 norm of dL/dh_t at each
                   timestep BEFORE it gets added to the incoming gradient from
                   t+1. This is what you'll plot to see vanishing gradients —
                   keep this instrumentation, don't skip it.

        TODO(2): Implement backprop through time.

        Cross-entropy + softmax gradient (standard result, you may use
        directly): if p_t is the softmax output and target is the true class
        index, then
            dL/dy_t = p_t.copy(); dL/dy_t[target] -= 1

        #Amr: dL/dy_t = (p_t - y_target) >> y_target is one-hot truth label
        this means that the target index of y_t will be substracted by 1 and other indecies remain the same

        Then for each parameter:
            dL/dW_hy += outer(dL/dy_t, h_t)
            dL/db_y  += dL/dy_t

        The hard part — backprop through the recurrence — must go in REVERSE
        time order (t = T-1 down to 0), accumulating a running
        `dh_next` term that represents dL/dh_t contributed by everything at
        timesteps > t:

            dh_t = W_hy.T @ dL/dy_t  +  dh_next        (dh_next=0 at t=T-1 start)
            da_t = dh_t * (1 - h_t**2)                 (tanh derivative; h_t = tanh(a_t))
            dW_xh += outer(da_t, x_t)
            dW_hh += outer(da_t, h_{t-1})
            db_h  += da_t
            dh_next = W_hh.T @ da_t                    (this becomes dh_next for t-1)

        Record ||dh_t|| (before adding dh_next... or after — try both and
        think about which one is the more honest measure of "gradient signal
        strength at this timestep", and note your reasoning in your writeup)
        into grad_norms_per_t.

        This is exactly where you should observe: as T grows, da_t and dh_t
        for early timesteps shrink toward zero (vanishing) or blow up
        (exploding), because dh_next = W_hh.T @ da_t is applied T times
        multiplicatively, once per timestep, with a tanh-derivative factor
        (which is in [0,1] and often << 1) at every step.
        """
        grads = {
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "b_h": np.zeros_like(self.b_h),
            "W_hy": np.zeros_like(self.W_hy),
            "b_y": np.zeros_like(self.b_y),
        }

        T = len(xs)
        grad_norms_per_t = [0.0] * T

        # TODO(2): implement the reverse-time loop described above
         
        # Gradient arriving from future recurrent timesteps.
        # At the final timestep, there is no future hidden state. 
        dh_next = np.zeros_like(hs[0])

        for t in range(T-1, -1, -1):
            h_t = hs[t+1]
            h_prev = hs[t]
            dh_t = dh_next.copy()

            target = ys_target[t]
            if target is not None: 
                # gradient of the softmax output
                dy_t = ps[t].copy()
                dy_t[target] -= 1.0
                # gradients through y_t = W_hy @ h_t + b_y
                grads["W_hy"] += np.outer(dy_t, h_t)
                grads["b_y"] += dy_t

                #local output loss gradient to h_t
                dh_t += self.W_hy.T @ dy_t # it already has a copy of dh_next
                
            #Now the local dL/dh_t = local output contribution + future recurrent contribution
            grad_norms_per_t[t] = np.linalg.norm(dh_t)
            # da_t = derivative of hidden state * derivative of hidden state with respect to activation
            da_t = dh_t * (1 - h_t**2)

            # Backpropagate through:
            # a_t = W_xh @ x_t + W_hh @ h_prev + b_h
            grads["W_xh"] += np.outer(da_t, xs[t])
            grads["W_hh"] += np.outer(da_t, h_prev)
            grads["b_h"] += da_t

            # send grads to the previous timestamp
            dh_next = self.W_hh.T @ da_t #dL/dh = dL/da * da/dh = W_hh * h_t*(1-h_t) 

            
        return grads, grad_norms_per_t

    # ------------------------------------------------------------------
    # Utility: gradient clipping (LSTM/GRU need this less, RNN needs it a lot —
    # notice why once you've implemented backward())
    # ------------------------------------------------------------------
    @staticmethod
    def clip_grads(grads, max_norm=5.0):
        total_norm = np.sqrt(
            sum(np.sum(grad**2) for grad in grads.values())
        )

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-12)

            for key in grads:
                grads[key] *= scale

        return grads

    def sgd_step(self, grads, lr=0.1):
        self.W_xh -= lr * grads["W_xh"]
        self.W_hh -= lr * grads["W_hh"]
        self.b_h -= lr * grads["b_h"]
        self.W_hy -= lr * grads["W_hy"]
        self.b_y -= lr * grads["b_y"]


# ----------------------------------------------------------------------
# Numerical gradient check — run this after implementing forward+backward
# to verify your BPTT is correct BEFORE trusting any experiment built on it.
# This is standard practice, not optional scaffolding.
# ----------------------------------------------------------------------
def gradient_check(model, xs, ys_target, epsilon=1e-5, num_checks=20, seed=0):
    """
    Compares analytic gradients (from backward()) against numerical gradients
    (finite differences on the loss) for a random sample of parameter entries.
    Relative error should be < 1e-4 for a correct implementation.
    """
    rng = np.random.default_rng(seed)

    def compute_loss(ys_target, ps):
        loss = 0.0
        for t, target in enumerate(ys_target):
            if target is not None:
                loss -= np.log(ps[t][target] + 1e-12)
        return loss

    ys, ps, hs = model.forward(xs)
    grads, _ = model.backward(xs, ys_target, ps, hs)

    param_names = ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]
    for name in param_names:
        param = getattr(model, name)
        analytic_grad = grads[name]
        flat_idx = rng.choice(param.size, size=min(num_checks, param.size), replace=False)
        for idx in flat_idx:
            multi_idx = np.unravel_index(idx, param.shape)
            old_val = param[multi_idx]

            param[multi_idx] = old_val + epsilon
            _, ps_plus, _ = model.forward(xs)
            loss_plus = compute_loss(ys_target, ps_plus)

            param[multi_idx] = old_val - epsilon
            _, ps_minus, _ = model.forward(xs)
            loss_minus = compute_loss(ys_target, ps_minus)

            param[multi_idx] = old_val  # restore

            numeric_grad = (loss_plus - loss_minus) / (2 * epsilon)
            analytic = analytic_grad[multi_idx]
            rel_error = abs(numeric_grad - analytic) / (abs(numeric_grad) + abs(analytic) + 1e-12)
            status = "OK" if rel_error < 1e-4 else "MISMATCH"
            print(f"{name}{multi_idx}: analytic={analytic:.6f} numeric={numeric_grad:.6f} "
                  f"rel_error={rel_error:.2e} [{status}]")


if __name__ == "__main__":
    # Tiny smoke test once you've implemented forward/backward.
    input_size, hidden_size, output_size = 5, 8, 5
    T = 6
    rng = np.random.default_rng(0)

    model = VanillaRNN(input_size, hidden_size, output_size, seed=0)
    xs = []
    for _ in range(T):
        vec = np.zeros(input_size)
        vec[rng.integers(input_size)] = 1.0
        xs.append(vec)
    ys_target = [rng.integers(output_size) for _ in range(T)]

    gradient_check(model, xs, ys_target)
