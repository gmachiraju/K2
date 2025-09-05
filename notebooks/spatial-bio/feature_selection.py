from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from utils import serialize, deserialize
import os

def top_positives_threshold_mass(vector, threshold=0.9):
    """
    Select the top positive values that contribute to the given probability threshold
    of the total positive mass, while preserving associated string labels.

    Parameters:
    - vector (list of tuples): List of (label, value) pairs.
    - threshold (float): Fraction of total positive mass to retain (default is 0.9).

    Returns:
    - list of tuples: The top (label, value) pairs contributing to the given mass threshold.
    """
    # Convert input to NumPy array for easy processing
    labels, values = zip(*vector)  # Separate labels and values
    values = np.array(values)      # Convert values to a NumPy array

    # Filter only positive values and corresponding labels
    positive_mask = values > 0
    positives = values[positive_mask]
    positive_labels = np.array(labels)[positive_mask]

    if positives.size == 0:
        return []  # Return empty list if no positive values
    
    # Sort indices by descending values
    sorted_indices = np.argsort(-positives)  # Negative sign for descending order
    sorted_positives = positives[sorted_indices]
    sorted_labels = positive_labels[sorted_indices]

    # Compute cumulative sum
    cumsum = np.cumsum(sorted_positives)

    # Compute the total sum of positive values
    total_positive_mass = cumsum[-1]

    # Find the index where cumulative sum exceeds the specified mass threshold
    cutoff_mass = threshold * total_positive_mass
    cutoff_index = np.searchsorted(cumsum, cutoff_mass, side='right')

    # Return the top (label, value) pairs contributing to the given mass threshold
    return list(zip(sorted_labels[:cutoff_index + 1], sorted_positives[:cutoff_index + 1]))


def compute_aic(model, X, y):
    """
    Computes the AIC for a fitted sklearn LogisticRegression model.
    
    Parameters:
    - model: fitted LogisticRegression model
    - X: input features
    - y: true binary labels
    
    Returns:
    - AIC score
    """
    # Predicted probabilities
    prob = model.predict_proba(X)
    
    # Log-likelihood (note: log_loss returns negative log-likelihood averaged per sample)
    log_likelihood = -log_loss(y, prob, normalize=False)
    
    # Number of parameters (coefficients + intercept)
    k = model.coef_.size + model.intercept_.size
    
    aic = 2 * k - 2 * log_likelihood
    return aic


def compute_aic_ovr(ovr_model, X, y):
    """
    Computes AIC for each binary classifier in a fitted OneVsRestClassifier.
    
    Parameters:
    - ovr_model: fitted OneVsRestClassifier model
    - X: input features
    - y: true multiclass labels
    
    Returns:
    - Dictionary mapping class index to its AIC
    """
    aic_per_class = {}
    classes = ovr_model.classes_

    for i, binary_model in enumerate(ovr_model.estimators_):
        # Create binary labels for class i vs rest
        c = classes[i]
        y_binary = (y == c).astype(int)
        
        # Predicted probabilities for class i
        prob = binary_model.predict_proba(X)
        
        # Log-likelihood
        try:
            log_likelihood = -log_loss(y_binary, prob, normalize=False)
        except ValueError:
            aic_per_class[c] = np.nan
            continue
        
        # Number of parameters (coefficients + intercept)
        k = binary_model.coef_.size + binary_model.intercept_.size
        
        # AIC for this binary model
        aic = 2 * k - 2 * log_likelihood
        aic_per_class[c] = aic

    return aic_per_class


def compute_aic_by_threshold(estimator, X, y, feature_names, thresholds=[0.2, 0.3, 0.4] + list(np.linspace(0.5, 0.9, 5)) + list(np.linspace(0.91, 1.0, 10))):
    """
    Compute AICs over various positive coefficient mass thresholds for a single estimator.

    Returns:
    - list of dicts with keys: threshold, k, aic
    """
    coef = estimator.coef_[0]
    feature_vector = list(zip(feature_names, coef))

    results = []

    for thresh in thresholds:
        selected = top_positives_threshold_mass(feature_vector, threshold=thresh)
        selected_features = [label for label, _ in selected]

        if not selected_features:
            results.append({"threshold": thresh, "k": 0, "aic": np.nan})
            continue

        # Subset X
        if isinstance(X, np.ndarray):
            selected_indices = [feature_names.index(f) for f in selected_features]
            X_subset = X[:, selected_indices]
        else:
            X_subset = X[selected_features]

        # Fit new model on the selected features
        model = LogisticRegression(max_iter=1000)
        model.fit(X_subset, y)

        prob = model.predict_proba(X_subset)
        log_likelihood = -log_loss(y, prob, normalize=False)
        k = model.coef_.size + model.intercept_.size
        aic = 2 * k - 2 * log_likelihood

        results.append({"threshold": thresh, "k": len(selected_features), "aic": aic})

    return results


def plot_aic_k_threshold(results):
    """
    Plots AIC vs number of features (k), with mass threshold on the secondary y-axis.
    
    Parameters:
    - results: list of dicts with keys 'threshold', 'k', and 'aic'
    """
    ks = [r['k'] for r in results]
    aics = [r['aic'] for r in results]
    thresholds = [r['threshold'] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Features (k)')
    ax1.set_ylabel('AIC', color=color1)
    ax1.plot(ks, aics, marker='o', color=color1, label='AIC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Secondary y-axis for threshold
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Mass Threshold', color=color2)
    ax2.plot(ks, thresholds, marker='x', linestyle='--', color=color2, label='Threshold')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("AIC vs Number of Features with Mass Threshold Overlay")
    fig.tight_layout()
    plt.show()


def plot_aic_threshold_k(results):
    """
    Plots AIC vs Mass Threshold, with number of features (k) on the secondary y-axis.

    Parameters:
    - results: list of dicts with keys 'threshold', 'k', and 'aic'
    """
    thresholds = [r['threshold'] for r in results]
    aics = [r['aic'] for r in results]
    ks = [r['k'] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Mass Threshold')
    ax1.set_ylabel('AIC', color=color1)
    ax1.plot(thresholds, aics, marker='o', color=color1, label='AIC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Secondary y-axis for k
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Number of Features (k)', color=color2)
    ax2.plot(thresholds, ks, marker='x', linestyle='--', color=color2, label='k')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("AIC vs Mass Threshold with Feature Count Overlay")
    fig.tight_layout()
    plt.show()


def plot_max_aic_per_k(aggregated_results):
    """
    Plots maximum AIC vs number of features (k), with associated threshold on right y-axis.

    Parameters:
    - aggregated_results: list of dicts from aggregate_max_aic_per_k()
    """
    ks = [r["k"] for r in aggregated_results]
    aics = [r["aic"] for r in aggregated_results]
    thresholds = [r["threshold"] for r in aggregated_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel("Number of Features (k)")
    ax1.set_ylabel("Max AIC", color=color1)
    ax1.plot(ks, aics, marker='o', color=color1, label="Max AIC")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel("Mass Threshold (at max AIC)", color=color2)
    ax2.plot(ks, thresholds, marker='x', linestyle='--', color=color2, label="Threshold")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Max AIC vs Number of Features with Threshold Overlay")
    fig.tight_layout()
    plt.show()


def plot_max_aic_per_k_rescaled(aggregated_results):
    """
    Plots linearly rescaled AIC (0–1) vs number of features (k),
    with threshold on the secondary y-axis.

    Parameters:
    - aggregated_results: list of dicts with 'k', 'aic', 'threshold'
    """
    ks = [r["k"] for r in aggregated_results]
    aics = [r["aic"] for r in aggregated_results]
    thresholds = [r["threshold"] for r in aggregated_results]

    # Min-max scale AICs
    aic_min = np.min(aics)
    aic_max = np.max(aics)
    aic_scaled = [(a - aic_min) / (aic_max - aic_min) if aic_max != aic_min else 0.0 for a in aics]

    fig, ax1 = plt.subplots(figsize=(6, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel("Number of Features (k)")
    ax1.set_ylabel("Rescaled AIC (0 = best, 1 = worst)", color=color1)
    ax1.plot(ks, aic_scaled, marker='o', color=color1, label="Rescaled AIC")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)

    # Secondary y-axis for threshold
    color2 = 'tab:green'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mass Threshold", color=color2)
    ax2.plot(ks, thresholds, marker='x', linestyle='--', color=color2, label="Threshold")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.1, 1.1)

    plt.title("Linearly Rescaled AIC vs Number of Features with Threshold Overlay")
    fig.tight_layout()
    plt.show()


def plot_max_aic_per_k_rescaled_elbowed(aggregated_results, plot_title=None):
    """
    Plots linearly rescaled AIC vs number of features (k), with mass threshold on secondary y-axis,
    and automatically selects a k using elbow detection + mass threshold > 0.5.

    Parameters:
    - aggregated_results: list of dicts with 'k', 'aic', 'threshold'
    """
    ks = [r["k"] for r in aggregated_results]
    aics = [r["aic"] for r in aggregated_results]
    thresholds = [r["threshold"] for r in aggregated_results]

    # Rescale AICs to 0–1 range
    aic_min = np.min(aics)
    aic_max = np.max(aics)
    aic_scaled = [(a - aic_min) / (aic_max - aic_min) if aic_max != aic_min else 0.0 for a in aics]

    # Step 1: Only consider indices where threshold > 0.5
    valid_idxs = [i for i, t in enumerate(thresholds) if t > 0.5]
    elbow_k = None

    if valid_idxs:
        # Step 2: Among those, compute slopes of AIC vs k
        slopes = []
        for i in range(1, len(valid_idxs)):
            i_prev = valid_idxs[i - 1]
            i_curr = valid_idxs[i]
            delta_aic = aic_scaled[i_curr] - aic_scaled[i_prev]
            delta_k = ks[i_curr] - ks[i_prev]
            if delta_k != 0:
                slope = delta_aic / delta_k
                slopes.append((slope, i_curr))  # store slope and "after the drop" index

        # Step 3: Find first slope < -1 or the steepest descent
        elbow_index = None
        strong_slope_indices = [s for s in slopes if s[0] < -1]

        if strong_slope_indices:
            elbow_index = strong_slope_indices[0][1]
        elif slopes:
            # fallback: most negative slope overall
            elbow_index = min(slopes, key=lambda x: x[0])[1]

        if elbow_index is not None:
            elbow_k = ks[elbow_index]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(6, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel("Number of Features (k)")
    ax1.set_ylabel("Rescaled AIC (0 = best, 1 = worst)", color=color1)
    ax1.plot(ks, aic_scaled, marker='o', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)

    # Secondary y-axis for threshold
    color2 = 'tab:green'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mass Threshold", color=color2)
    ax2.plot(ks, thresholds, marker='x', linestyle='--', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.1, 1.1)

    # Draw vertical line at elbow_k
    if elbow_k is not None:
        ax1.axvline(x=elbow_k, color='purple', linestyle=':', linewidth=2)
        # label=f"Selected k = {elbow_k}"

    if plot_title is None:
        plt.title("AIC Elbow Analysis for Group Feature Selection")
    else:
        plt.title(plot_title)
    fig.tight_layout()
    # if elbow_k is not None:
    #     ax1.legend(loc='upper right')
    plt.show()
    return elbow_k  # Return selected k if useful downstream


# cumulative importance thresholding
def find_importance_threshold(coefs):

    # Step 1: Get absolute coefficients
    coefs = np.abs(coefs).flatten()  # for binary classification

    # Step 2: Normalize (L1 or softmax)
    # Option 1: Softmax
    softmax = np.exp(coefs) / np.sum(np.exp(coefs))

    # Option 2: L1 normalize (simpler)
    l1_norm = coefs / np.sum(coefs)

    # Choose either softmax or l1_norm
    importance = l1_norm

    # Step 3: Rank and sort
    sorted_idx = np.argsort(-importance)
    sorted_importance = importance[sorted_idx]

    # Step 4: Cumulative sum
    cumulative = np.cumsum(sorted_importance)

    # Optional: Plot to visually find the elbow
    plt.figure(figsize=(8,4))
    plt.plot(cumulative, marker='o')
    plt.title('Cumulative Importance of Features')
    plt.xlabel('Sorted Feature Rank')
    plt.ylabel('Cumulative Importance')
    plt.grid(True)
    plt.show()


def analyze_feature_importance(coefs, feature_names, method='l1', threshold=0.8, plot=True):
    """
    Analyzes feature importance from a fitted logistic regression model.

    Parameters:
        model: Fitted sklearn LogisticRegression model.
        feature_names: List of feature names corresponding to the model's input features.
        method: 'l1' or 'softmax' to choose normalization method.
        threshold: Cumulative importance threshold (e.g., 0.8).
        plot: If True, displays a plot of cumulative feature importance.

    Returns:
        selected_features: List of feature names selected by the threshold.
    """
    # Step 1: Get absolute coefficients
    coefs = np.abs(coefs).flatten()  # for binary classification

    # Step 2: Normalize
    if method == 'softmax':
        importance = np.exp(coefs) / np.sum(np.exp(coefs))
    elif method == 'l1':
        importance = coefs / np.sum(coefs)
    else:
        raise ValueError("Normalization method must be 'l1' or 'softmax'")

    # Step 3: Rank and sort
    sorted_idx = np.argsort(-importance)
    sorted_importance = importance[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]

    # Step 4: Cumulative sum
    cumulative = np.cumsum(sorted_importance)

    # Step 5: Find threshold index
    threshold_idx = np.argmax(cumulative >= threshold)
    selected_features = sorted_features[:threshold_idx + 1].tolist()

    # Optional plot
    if plot:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(sorted_features)), sorted_importance)
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
        plt.title('Feature Importance ({}-normalized)'.format(method))
        plt.xlabel('Features (sorted by importance)')
        plt.ylabel('Normalized Importance')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(cumulative, marker='o')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
        plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
        plt.title('Cumulative Importance of Features')
        plt.xlabel('Sorted Feature')
        plt.ylabel('Cumulative Importance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_features


def find_elbow(cumulative):
    diffs = np.diff(cumulative)
    second_diffs = np.diff(diffs)
    return np.argmax(second_diffs) + 2  # +2 to adjust for two diffs


def analyze_feature_importance_positives(coefs, feature_names, method='l1', threshold=0.8, plot=True):
    """
    Analyzes feature importance from a fitted logistic regression model,
    considering only positive coefficients (class-1 selectors).

    Parameters:
        model: Fitted sklearn LogisticRegression model.
        feature_names: List of feature names corresponding to the model's input features.
        method: 'l1' or 'softmax' to choose normalization method.
        threshold: Cumulative importance threshold (e.g., 0.8).
        plot: If True, displays a plot of cumulative feature importance.

    Returns:
        selected_features: List of feature names selected by the threshold.
    """
    # Step 1: Get positive coefficients only
    coefs = coefs.flatten()
    positive_mask = coefs > 0
    positive_coefs = coefs[positive_mask]
    positive_features = np.array(feature_names)[positive_mask]

    # Step 2: Normalize
    if method == 'softmax':
        importance = np.exp(positive_coefs) / np.sum(np.exp(positive_coefs))
    elif method == 'l1':
        importance = positive_coefs / np.sum(positive_coefs)
    else:
        raise ValueError("Normalization method must be 'l1' or 'softmax'")

    # Step 3: Rank and sort
    sorted_idx = np.argsort(-importance)
    sorted_importance = importance[sorted_idx]
    sorted_features = positive_features[sorted_idx]

    # Step 4: Cumulative sum
    cumulative = np.cumsum(sorted_importance)

    # Step 5: Threshold and elbow detection
    threshold_idx = np.argmax(cumulative >= threshold)

    diffs = np.diff(cumulative)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2  # Adjust for second derivative shift

    selected_features = sorted_features[:threshold_idx + 1].tolist()

    # Optional plot
    if plot:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(sorted_features)), sorted_importance)
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
        plt.axvline(threshold_idx, color='r', linestyle='--', label='Threshold Index')
        plt.axvline(elbow_idx, color='g', linestyle='--', label='Elbow Index')
        plt.title('Positive Feature Importance ({}-normalized)'.format(method))
        plt.xlabel('Features (sorted by importance)')
        plt.ylabel('Normalized Importance')
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(cumulative, marker='o')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
        plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
        plt.axhline(cumulative[elbow_idx], color='g', linestyle='--', label='Elbow Value')
        plt.title('Cumulative Importance of Positive Coefficients')
        plt.xlabel('Sorted Feature')
        plt.ylabel('Cumulative Importance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_features


def aggregate_max_aic_per_k(results):
    """
    Aggregate results to get the maximum AIC for each unique k.

    Parameters:
    - results: list of dicts with keys 'threshold', 'k', and 'aic'

    Returns:
    - list of dicts with keys 'k', 'aic', 'threshold' (corresponding to max AIC)
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        if np.isnan(r["aic"]):
            continue
        grouped[r["k"]].append(r)

    aggregated = []
    for k_val, entries in grouped.items():
        max_entry = max(entries, key=lambda r: r["aic"])
        aggregated.append({
            "k": k_val,
            "aic": max_entry["aic"],
            "threshold": max_entry["threshold"]
        })

    # Sort by increasing k
    return sorted(aggregated, key=lambda r: r["k"])


def grab_viz_coefficients(concept_idx, clf, all_feats):
    plt.figure(figsize=(3, 10))
    to_pull = concept_idx + 1
    b = clf.estimators_[to_pull].coef_[0] # importance weights
    plt.barh(all_feats, b, color="k")
    plt.title("ElasticNet model coefficients")
    plt.show()
    return b


def run_full_aic_analysis(concept_idx, clf, all_feats, Xt, yt_multi):
    b = grab_viz_coefficients(concept_idx, clf, all_feats)
    scores = dict(zip(all_feats, b))
    out = list({k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}.items())[0:5]
    print("\npositive weights:")
    print(out)
    sorted_scores = list({k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}.items())
    out = top_positives_threshold_mass(sorted_scores, threshold=0.7)
    print(out)

    print("\nvisualizing positive weights:")
    out = analyze_feature_importance_positives(b, all_feats, method='l1', threshold=0.7, plot=True)
    print(out)

    print("\nAIC analysis:")
    cache_path = f"/oak/stanford/groups/paragm/gautam/prospection/K2/notebooks/spatial-bio/aic_concept{concept_idx}.obj"
    if os.path.exists(cache_path):
        print("results already cached, pulling now...")
        aic_results = deserialize(cache_path)
    else:
        print("results missing, rerunning now...")
        estimator = clf.estimators_[concept_idx+1]
        aic_results = compute_aic_by_threshold(estimator, Xt, yt_multi, all_feats)
        serialize(aic_results, cache_path)

    # plot_aic_k_threshold(aic_results)
    agg_results = aggregate_max_aic_per_k(aic_results)
    # plot_max_aic_per_k(agg_results)
    plot_max_aic_per_k_rescaled(agg_results)
    if concept_idx == -1:
        plot_title = "Feature Selection for Non-salient Cell Group"
    else:
        plot_title = f"Feature Selection for Cell Group {concept_idx}"
    selected_k = plot_max_aic_per_k_rescaled_elbowed(agg_results, plot_title)
    print("Automatically selected k:", selected_k)