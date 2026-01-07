import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from zennit.attribution import IntegratedGradients
from zennit.attribution import Gradient
from zennit.rules import Gamma
from zennit.composites import LayerMapComposite

# --------------------------
# Base model training
# --------------------------
def train_base(model, X_train, X_val, y_train, y_val, num_epochs, lr=0.001, print_every=0, weight_decay=1e-10):
    """
    Train a base PyTorch model using MSE loss and Adam optimizer.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_outputs = model(X_val)
        val_loss = criterion(val_outputs.flatten(), y_val.flatten())

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if print_every > 0 and (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            fig, ax = plt.subplots(ncols=2, figsize=(5,2.5))
            ax[0].plot(train_losses, c='blue', marker='X')
            ax[0].plot(val_losses, c='orange', marker='o')
            ax[0].set_xlabel('Iteration')
            ax[0].set_ylabel('Training Error')
            ax[0].set_title('Training Error Over Iterations')

            ax[1].scatter(y_train, model.cpu()(torch.Tensor(X_train)).detach().cpu().numpy(), c='grey' , ec='k',alpha=0.25)
            ax[1].plot([0,1],[0,1], c='darkred')
            ax[1].set_xlabel('y_true')
            ax[1].set_ylabel('y_pred')
            ax[1].set_title('True vs Predicted')

            plt.tight_layout()
            plt.show()

    return model


# --------------------------
# Train Xpert heads
# --------------------------
def train_heads(model, X_train, y_train, lst_ranges_heads=None, layer_freeze_head=6, n_neurons_heads=20, lr=1e-4, base_steps=2500, weight_decay=1e-10, plot=True):
    """
    Train Xpert heads over disjoint target ranges.
    """
    if lst_ranges_heads is None:
        lst_ranges_heads = list(np.arange(0, 1.01, 0.33))

    X_tensor = torch.Tensor(X_train).cpu()
    y_tensor = torch.Tensor(y_train).cpu()
    model = model.cpu()
    lst_heads = []

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    for i_range, low in enumerate(lst_ranges_heads[:-1]):
        high = lst_ranges_heads[i_range + 1]
        print(f"Training head for range [{low:.2f}, {high:.2f}]")

        mask_range = ((y_tensor > low) & (y_tensor < high)).flatten()
        y_residual = np.clip(y_pred - low, 0, high - low)
        head_model = copy.deepcopy(model)

        if i_range > 0:
            for idx_layer, param in enumerate(head_model.parameters()):
                if idx_layer < layer_freeze_head:
                    param.requires_grad = False

            head_model = rescale_top(copy.deepcopy(head_model))
            with torch.no_grad():
                activations = head_model[:-1](X_tensor[mask_range])

            U, S, V = torch.pca_lowrank(activations, q=n_neurons_heads, center=False, niter=50)
            head_model = extend_top(head_model, n_neurons_heads, V)

            head_model = train_base(
                head_model,
                X_tensor[mask_range],
                X_train,
                y_residual[mask_range],
                y_residual[:len(X_train)],
                base_steps * i_range,
                lr=lr,
                print_every=1000,
                weight_decay=weight_decay
            )

        head_model = add_rectifier(head_model, high - low)

        if plot:
            with torch.no_grad():
                y_model = head_model(X_tensor).cpu().numpy()
            plt.figure(figsize=(5,2.5))
            plt.scatter(y_pred, y_model + low, alpha=0.8, color='blue', label="Xpert output")
            plt.scatter(y_pred, y_pred, alpha=0.3, color='orange', label="Original output")
            plt.xlabel("y_true")
            plt.ylabel("y_model")
            plt.title("Xpert vs Original")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        lst_heads.append(head_model)

    return lst_heads


# --------------------------
# Metrics
# --------------------------
def pearson_corrcoef(tensor1, tensor2):
    """
    Compute Pearson correlation coefficient between two tensors.
    """
    mean1, mean2 = torch.mean(tensor1), torch.mean(tensor2)
    cov = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
    std1, std2 = torch.std(tensor1), torch.std(tensor2)
    return (cov / (std1 * std2)).item()


# --------------------------
# Plotting functions
# --------------------------
def plot_wine_regression_overview(model, X_train, y_train, lst_heads, corr_coef, save_path=None, device='cpu'):
    """
    Overview plot: target distribution, model vs true, expert heads.
    """
    X_tensor = torch.Tensor(X_train).to(device)
    model = model.to(device)

    with torch.no_grad():
        y_hat = model(X_tensor).cpu().numpy()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(ncols=3, figsize=(10,3))
    fig.suptitle("Wine Regression Problem")

    ax[0].axvspan(0.75, 1.0, color="red", alpha=0.5, label="top wines")
    bins = np.arange(0, 1.01, 0.05)
    ax[0].hist(y_hat[y_hat < 0.33], bins=bins, color=cycle[0], ec="k", alpha=0.8, label="low")
    ax[0].hist(y_hat[y_hat > 0.33], bins=bins, color=cycle[1], ec="k", alpha=0.8, label="mid")
    ax[0].hist(y_hat[y_hat > 0.66], bins=bins, color=cycle[2], ec="k", alpha=0.8, label="high")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel("# samples")
    ax[0].set_title("Target distribution")
    ax[0].legend()

    ax[1].scatter(y_train, y_hat, c="grey", ec="k", alpha=0.25)
    ax[1].plot([0,1],[0,1], c="darkred", lw=2)
    ax[1].set_xlabel("y_true")
    ax[1].set_ylabel("y_hat")
    ax[1].set_title(f"Model performance (R^2={corr_coef:.2f})")

    for head in lst_heads:
        head = head.to(device)
        with torch.no_grad():
            y_expert = head(X_tensor).cpu().numpy()
        ax[2].scatter(y_hat, y_expert, alpha=0.25, ec="k")

    ax[2].set_xlabel("y_hat")
    ax[2].set_ylabel("y_expert")
    ax[2].set_title("XpertAI-head performance")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


def plot_wine_attributions(model, X_train, att_naive, lst_att_heads, lst_heads, input_names, weights=(0.33,0.33,0.125), top_threshold=0.75, save_path=None):
    """
    Plot mean feature attributions for high-quality wines.
    """
    X_tensor = torch.Tensor(X_train)
    with torch.no_grad():
        outputs = model(X_tensor).cpu().numpy().flatten()

    mask_top = outputs > top_threshold

    fig, ax = plt.subplots(figsize=(6,4))

    # Naive model
    ax.bar(
        np.arange(att_naive.shape[1]),
        att_naive[mask_top].mean(axis=0),
        yerr=att_naive[mask_top].std(axis=0),
        width=0.6,
        color="grey",
        edgecolor="k",
        alpha=0.5,
        error_kw=dict(ecolor="grey"),
        label="NAIVE"
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, att in enumerate(lst_att_heads):
        mean_att = att[mask_top].mean(axis=0)
        std_att = att[mask_top].std(axis=0)
        norm = mean_att.sum() if mean_att.sum() != 0 else 1.0

        ax.bar(
            np.arange(att.shape[1]) - 0.2 + j*0.2,
            mean_att / norm * weights[j],
            yerr=std_att / norm * weights[j],
            width=0.2,
            color=cycle[j],
            edgecolor="k",
            label=f"Xpert_{['low','mid','high'][j]}"
        )

    ax.set_xticks(np.arange(len(input_names)))
    ax.set_xticklabels(input_names, rotation=45, weight="bold")
    ax.set_ylabel("Mean Attributions")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


# --------------------------
# Helper layers / heads
# --------------------------
def rescale_top(net):
    """Rescale top Linear->ReLU->Linear block to ±1 final weights."""
    with torch.no_grad():
        w1_new = nn.Parameter(torch.matmul(torch.diag_embed(torch.abs(net[-1].weight)).squeeze(), net[-3].weight), requires_grad=True)
        b1_new = nn.Parameter(net[-3].bias * torch.abs(net[-1].weight).squeeze(), requires_grad=True)
        w2_new = nn.Parameter(torch.sign(net[-1].weight), requires_grad=True)

        net[-3].weight = nn.Parameter(w1_new)
        net[-3].bias = nn.Parameter(b1_new.reshape(1, net[-3].bias.shape[0]))
        net[-1].weight = nn.Parameter(w2_new)
    return net


def extend_top(net, n_pc, V):
    """Insert fixed linear projection and its inverse before final layer."""
    layers = list(net[:-1]) + [nn.Linear(net[-1].in_features, n_pc, bias=False),
                               nn.Linear(n_pc, net[-1].in_features, bias=False),
                               net[-1]]
    net_new = nn.Sequential(*layers)
    with torch.no_grad():
        net_new[-3].weight = nn.Parameter(V.T.clone(), requires_grad=False)
        net_new[-2].weight = nn.Parameter(V.clone(), requires_grad=False)
    return net_new


def add_rectifier(model_head, upper_cap):
    """Append fixed rectifier block clamping output to [0, upper_cap]."""
    layers = [nn.ReLU(),
              nn.Linear(1,2, bias=True),
              nn.ReLU(),
              nn.Linear(2,1,bias=False)]
    with torch.no_grad():
        layers[1].weight = nn.Parameter(torch.Tensor([1,1]).view(-1,1), requires_grad=False)
        layers[1].bias = nn.Parameter(torch.Tensor([0,-upper_cap]), requires_grad=False)
        layers[3].weight = nn.Parameter(torch.Tensor([1,-1]).view(1,-1), requires_grad=False)
    return nn.Sequential(*model_head, *layers)


# --------------------------
# Explain and evaluate
# --------------------------
def calculate_attributions(model, attributor, inputs):
    """Compute feature attributions for a model and given inputs."""
    with torch.no_grad():
        outputs = model.cpu()(inputs).cpu()
    with attributor as attr:
        _, attributions = attr(inputs.cpu(), outputs.cpu())
    return attributions

def flip_feats(model, x, baseline, attribution):
    """
    Perform feature flipping according to attribution scores.

    Returns outputs for descending and ascending flips.
    """
    model.eval()
    attribution = np.asarray(attribution)
    orders = {"desc": np.argsort(attribution)[::-1], "asc": np.argsort(attribution)}
    results = {}

    with torch.no_grad():
        for key, order in orders.items():
            x_flipped = x.clone()
            outputs = [model(x).item()]
            for idx in order:
                x_flipped[idx] = baseline[idx]
                outputs.append(model(x_flipped).item())
            results[key] = np.array(outputs)

    return results["desc"], results["asc"]

def evaluate_head_attributions(model, heads, X_train, ranges,
                               attribution_method="LRP", baseline_fn=None,
                               eval_top=350, n_baselines=10):
    """
    Evaluate naive vs. head-aggregated attributions using ABC
    (Area Between the Curves) via feature flipping.
    """
    def band_index(x, bands, side):
        idx = np.searchsorted(bands, x, side=side) - 1
        return int(np.clip(idx, 0, len(bands) - 2))

    model_cpu = model.cpu()
    inputs = torch.Tensor(X_train)

    # Attribution setup
    if attribution_method == "LRP":
        layer_map = [(nn.Linear, Gamma(0.2))]
        composite = LayerMapComposite(layer_map=layer_map)
        get_attr = lambda m: calculate_attributions(m, Gradient(m, composite), inputs)
    elif attribution_method == "IG":
        get_attr = lambda m: calculate_attributions(
            m, IntegratedGradients(m, n_iter=20, baseline_fn=baseline_fn), inputs)
    else:
        raise ValueError("attribution_method must be 'LRP' or 'IG'")

    att_base = get_attr(model_cpu)
    att_heads = [get_attr(h.cpu()) for h in heads]

    scores = model_cpu(inputs).flatten().detach().numpy()
    sorted_idx = np.argsort(scores)

    results = []
    abc_naive, abc_heads = [], []

    for rank in tqdm(range(eval_top)):
        sample_idx = sorted_idx[-rank]
        for _ in range(n_baselines):
            offset = np.random.randint(50, 400 - rank)
            base_idx = sorted_idx[max(-rank - offset, -400)]

            out_sample = model_cpu(inputs[sample_idx]).item()
            out_base = model_cpu(inputs[base_idx]).item()

            # Naive ABC
            fd, fu = flip_feats(model_cpu, inputs[sample_idx], inputs[base_idx], att_base[sample_idx])
            abc_n = np.mean(((fu - fu[-1]) - (fd - fu[-1])) / (fu[0] - fu[-1]))

            # Head-aggregated ABC
            start = band_index(out_base, ranges, side="right")
            stop = band_index(out_sample, ranges, side="left")

            att_combined = torch.zeros_like(att_base[sample_idx])
            for h in range(start, stop + 1):
                att_combined += att_heads[h][sample_idx]

            fd_h, fu_h = flip_feats(model_cpu, inputs[sample_idx], inputs[base_idx], att_combined)
            abc_h = np.mean(((fu_h - fu_h[-1]) - (fd_h - fu_h[-1])) / (fu_h[0] - fu_h[-1]))

            results.append({
                "out_in": out_sample,
                "out_base": out_base,
                "i_sample": sample_idx,
                "i_baseline": base_idx,
                "abc_naive": abc_n,
                "abc_heads": abc_h,
            })
            abc_naive.append(abc_n)
            abc_heads.append(abc_h)

    df = pd.DataFrame(results)
    print(f"average ABC naive: {np.round(np.mean(abc_naive), 2)}")
    print(f"average ABC XpertAI: {np.round(np.mean(abc_heads), 2)}")
    print(f"relative improvement: {np.round(((np.mean(abc_heads) / np.mean(abc_naive)) - 1) * 100)}%")

    return df, att_base, att_heads
