#!/usr/bin/env python3
"""
Demonstration of PreferenceOptimizer (model/preferenceOptimizer.py).

A simulated user has a hidden utility function over a 3D box (x, y, z offsets). The optimiser
never sees utility values: it only suggests two options, is told which one the user prefers,
and uses that to suggest the next two.

Run from anywhere:
    python examples/preference_optimizer_demo.py                # simulated user
    python examples/preference_optimizer_demo.py --noise 0.2    # user answers wrongly 20% of the time
    python examples/preference_optimizer_demo.py --human        # you answer the duels yourself
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr

from model.preferenceOptimizer import PreferenceOptimizer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--duels', type=int, default=15, help='number of duels in the loop')
parser.add_argument('--noise', type=float, default=0.0, help='probability that the simulated user answers wrongly')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--human', action='store_true', help='answer the duels yourself (type 1 or 2)')
parser.add_argument('--show', action='store_true', help='open the figures in a window as well as saving them')
args = parser.parse_args()

np.random.seed(args.seed)  # the samplers of the library use the global NumPy random state
rng = np.random.default_rng(args.seed)
fmt = lambda p: "[" + ", ".join(f"{v:+.3f}" for v in p) + "]"

def heading(text):
    print(f"\n=== {text} ===")


#####################################################################
# 1. Setup
#####################################################################
heading("1. Setup")
names = ['x', 'y', 'z']
bounds = np.array([[-0.10, 0.10],    # x offset [m]
                   [-0.15, 0.15],    # y offset [m]
                   [-0.05, 0.10]])   # z offset [m]
span = bounds[:, 1] - bounds[:, 0]

# hidden utility: a main peak and a lower secondary peak (centres given as fractions of the box)
main_peak = bounds[:, 0] + np.array([0.70, 0.35, 0.60]) * span
second_peak = bounds[:, 0] + np.array([0.25, 0.75, 0.30]) * span

def true_utility(points):
    u = (np.atleast_2d(points) - bounds[:, 0]) / span
    c1 = (main_peak - bounds[:, 0]) / span
    c2 = (second_peak - bounds[:, 0]) / span
    return (1.0 * np.exp(-np.sum((u - c1) ** 2, axis=1) / (2 * 0.25 ** 2))
            + 0.6 * np.exp(-np.sum((u - c2) ** 2, axis=1) / (2 * 0.20 ** 2)))

x_opt = minimize(lambda p: -true_utility(p)[0], main_peak, bounds=bounds).x
u_max = true_utility(x_opt)[0]
u_second = true_utility(minimize(lambda p: -true_utility(p)[0], second_peak, bounds=bounds).x)[0]

def ask_user(options):
    """Index (0 or 1) of the preferred option."""
    if args.human:
        for i, p in enumerate(options, start=1):
            print(f"   option {i}: {fmt(p)}")
        while True:
            answer = input("   Which do you prefer? Enter 1 or 2: ").strip()
            if answer in ('1', '2'):
                return int(answer) - 1
            print("   Please enter either 1 or 2")
    choice = int(np.argmax(true_utility(options)))
    if rng.random() < args.noise:
        choice = 1 - choice   # wrong answer
    return choice

opt = PreferenceOptimizer(bounds, q=2, seed=args.seed)
print("bounds:", ", ".join(f"{n} in [{lo:+.2f}, {hi:+.2f}]" for n, (lo, hi) in zip(names, bounds)))
if not args.human:
    print(f"hidden utility: main peak at {fmt(x_opt)} (utility {u_max:.2f}), "
          f"secondary peak near {fmt(second_peak)} (utility {u_second:.2f})")
    print(f"simulated user answers wrongly with probability {args.noise}")


#####################################################################
# 2. First query: no data yet, so the options are Sobol points
#####################################################################
heading("2. First query, before any preference")
options = opt.suggest()
print("suggest() ->", fmt(options[0]), "and", fmt(options[1]))
all_suggestions = [options]


#####################################################################
# 3. Duel loop: suggest -> ask -> add the preference
#####################################################################
heading(f"3. Duel loop ({args.duels} duels)")
best_utility = []
for duel in range(1, args.duels + 1):
    if duel > 1:
        options = opt.suggest()
        all_suggestions.append(options)
    choice = ask_user(options)
    opt.add_preferences(winners=options[choice], losers=options[1 - choice])

    best_point, _ = opt.best()
    line = f"duel {duel:2d}: {fmt(options[0])} vs {fmt(options[1])} -> prefers {choice + 1}"
    if not args.human:
        best_utility.append(true_utility(best_point)[0])
        line += f" | utility of best(): {best_utility[-1]:.2f} (max {u_max:.2f})"
    print(line)


#####################################################################
# 4. Several duels can be added in one call
#####################################################################
heading("4. Adding several duels in one call")
if args.human:
    print("skipped in --human mode")
else:
    extra = bounds[:, 0] + qmc.Sobol(d=3, scramble=True, seed=rng).random(4) * span
    order = np.argsort(-true_utility(extra))           # best first
    winners = extra[order[[0, 0, 0]]]                  # the best of the four beats the other three
    losers = extra[order[[1, 2, 3]]]
    opt.add_preferences(winners=winners, losers=losers)  # arrays of shape (3, 3)
    print(f"added {len(winners)} duels at once; the model now holds {len(opt.pairs)} duels "
          f"over {opt.X.shape[0]} points")
    best_utility.append(true_utility(opt.best()[0])[0])


#####################################################################
# 5. Reading the model
#####################################################################
heading("5. Reading the model")
best_point, best_mean = opt.best()
print(f"best()            -> {fmt(best_point)}, posterior mean {best_mean:.2f}")
query = np.vstack([best_point, bounds[:, 0]])
mu, sigma = opt.predict(query)
print(f"predict(best)     -> mean {mu[0]:+.2f}, std {sigma[0]:.2f}")
print(f"predict(corner)   -> mean {mu[1]:+.2f}, std {sigma[1]:.2f}   (lower corner of the box)")
model = opt.model   # the fitted erroneousPreference GP (its inputs are scaled to the unit cube)
print(f"opt.model.samples -> {model.samples.shape}  (compared points x posterior samples)")
print(f"opt.model.params  -> lengthscale {model.params['lengthscale']['value']}, "
      f"variance {model.params['variance']['value']}")
print(f"opt.X {opt.X.shape}, opt.pairs: {len(opt.pairs)} [winner, loser] index pairs")


#####################################################################
# 6. True against predicted utility
#####################################################################
heading("6. Figures")
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': '#c9c8c2',
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})

# file name suffix for the setting that was run: _human, _noise_<P>, or nothing for the standard run
suffix = '_human' if args.human else (f'_noise_{args.noise:g}' if args.noise > 0 else '')

centre = best_point if args.human else x_opt   # the slices go through this point
planes = [(0, 1), (0, 2), (1, 2)]
n_grid = 60
rows = ['predicted'] if args.human else ['true', 'predicted']
fig = plt.figure(figsize=(13, 4.4 * len(rows) + 0.9), layout='constrained')
subfigs = np.atleast_1d(fig.subfigures(len(rows), 1, hspace=0.06))
for r, kind in enumerate(rows):
    row_axes = subfigs[r].subplots(1, 3)
    subfigs[r].suptitle('True utility' if kind == 'true' else 'Predicted utility', x=0.01, ha='left',
                        fontsize=13, fontweight='bold', color=INK)
    for c, (a, b) in enumerate(planes):
        ax = row_axes[c]
        ga = np.linspace(*bounds[a], n_grid)
        gb = np.linspace(*bounds[b], n_grid)
        A, B = np.meshgrid(ga, gb)
        grid = np.tile(centre, (n_grid * n_grid, 1))
        grid[:, a], grid[:, b] = A.ravel(), B.ravel()
        values = true_utility(grid) if kind == 'true' else opt.predict(grid)[0]
        mesh = ax.pcolormesh(A, B, values.reshape(n_grid, n_grid), cmap='Blues', shading='auto')
        if kind == 'predicted':
            ax.scatter(opt.X[:, a], opt.X[:, b], s=22, color=INK, edgecolor='white', linewidth=0.8,
                       label='compared points (projected)')
            ax.scatter(best_point[a], best_point[b], s=110, marker='D', color=ORANGE, edgecolor='white',
                       linewidth=1.5, label='best()')
        if not args.human:
            ax.scatter(x_opt[a], x_opt[b], s=260, marker='*', color='white', edgecolor=INK, linewidth=1.2,
                       label='true optimum')
        ax.set_xlabel(f"{names[a]} [m]")
        ax.set_ylabel(f"{names[b]} [m]")
        other = ({0, 1, 2} - {a, b}).pop()
        ax.set_title(f"{names[a]}-{names[b]} plane at {names[other]} = {centre[other]:+.3f}", fontsize=10, color=MUTED)
        if c == 2:
            subfigs[r].colorbar(mesh, ax=row_axes, shrink=0.85, pad=0.015,
                         label='true utility' if kind == 'true' else 'predicted utility (posterior mean)')
handles, labels = row_axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
fig.suptitle(f"Utility after {len(opt.pairs)} duels. The colour scales differ: preferences fix the ordering "
             "of utilities, not their scale", fontsize=11, color=MUTED)
path_utility = os.path.join(REPO, 'examples', f'preference_optimizer_utility{suffix}.png')
fig.savefig(path_utility, dpi=130, bbox_inches='tight')
print("saved", path_utility)


#####################################################################
# 7. Progress and agreement (needs the true utility)
#####################################################################
if not args.human:
    test_points = bounds[:, 0] + qmc.Sobol(d=3, scramble=True, seed=rng).random(1024) * span
    test_true = true_utility(test_points)
    test_mean, _ = opt.predict(test_points)
    rho = spearmanr(test_mean, test_true)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    steps = np.arange(1, len(best_utility) + 1)
    ax1.plot(steps, best_utility, color=BLUE, linewidth=2, marker='o', markersize=5)
    ax1.axhline(u_max, color=MUTED, linestyle='--', linewidth=1)
    ax1.axhline(u_second, color=MUTED, linestyle=':', linewidth=1)
    ax1.text(steps[-1], u_max, 'true maximum', ha='right', va='bottom', color=MUTED)
    ax1.text(steps[-1], u_second, 'secondary peak', ha='right', va='bottom', color=MUTED)
    ax1.axvline(args.duels + 0.5, color='#c9c8c2', linewidth=1)
    ax1.text(args.duels + 0.5, 0.02, ' batch of 3 added', color=MUTED, ha='left', va='bottom', rotation=90, fontsize=8)
    ax1.set_ylim(0, u_max * 1.1)
    ax1.set_xlabel("update (one duel each, then the batch)")
    ax1.set_ylabel("true utility of best()")
    ax1.set_title("True utility of predicted best compared point", loc='left', fontweight='bold')
    ax1.grid(axis='y', color='#e6e5e0', linewidth=0.8)

    ax2.scatter(test_true, test_mean, s=9, color=BLUE, alpha=0.45, linewidth=0)
    ax2.set_xlabel("true utility")
    ax2.set_ylabel("predicted utility (posterior mean)")
    ax2.set_title(f"1024 test points, Spearman rank correlation {rho:.2f}", loc='left', fontweight='bold')
    ax2.grid(color='#e6e5e0', linewidth=0.8)
    fig.tight_layout()
    path_progress = os.path.join(REPO, 'examples', f'preference_optimizer_progress{suffix}.png')
    fig.savefig(path_progress, dpi=130, bbox_inches='tight')
    print("saved", path_progress)

    #####################################################################
    # Checks
    #####################################################################
    heading("Checks")
    S = np.vstack(all_suggestions)
    sd_points = opt.predict(opt.X)[1].mean()
    sd_corner = opt.predict(bounds[:, :1].T)[1][0]
    checks = [
        ("every suggestion lies inside the bounds", bool(np.all(S >= bounds[:, 0]) and np.all(S <= bounds[:, 1]))),
        ("opt.model.samples has one row per compared point", model.samples.shape[0] == opt.X.shape[0]),
        (f"utility of best() improved ({best_utility[0]:.2f} -> {best_utility[-1]:.2f})", best_utility[-1] > best_utility[0]),
        (f"predicted and true utility are positively rank-correlated ({rho:.2f})", rho > 0),
        (f"std is lower at compared points ({sd_points:.2f}) than at an unexplored corner ({sd_corner:.2f})", sd_points < sd_corner),
    ]
    for text, ok in checks:
        print("PASS" if ok else "FAIL", "-", text)

if args.show:
    plt.show()
