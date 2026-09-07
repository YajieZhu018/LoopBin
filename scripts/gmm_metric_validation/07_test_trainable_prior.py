### Tests the user's recalled prior finding: making u_p/lambda_p trainable (instead of
### frozen at their load_pretrained_weights init, the current default for every pi_mode)
### collapses training to one big cluster. Trains two VADE runs -- "frozen" (current
### default) vs "trainable" (loopbin.model.vade_model.GMM's trainable_prior=True, a
### diagnostic-only constructor kwarg, not wired to any CLI flag) -- from the *same*
### pinned GMM prior, same pretrained AE, same seed, so trainability is the only
### variable. See run_report.md's 2026-09-08 entry for the result.
import json
import os
import sys

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from loopbin.model.vade_model import VADE  # noqa: E402

DATA_PATH = os.path.join(REPO_ROOT, 'data/02_process/merged_log_data.npy')
PRETRAIN_PATH = os.path.join(REPO_ROOT,
    'scripts/gmm_metric_validation/results/real_latent_relu/pretrained_ae')
GMM_PRIOR_PATH = os.path.join(REPO_ROOT,
    'trials/vade_6clusters_merged_control_degron_rep1_relu_sep_pi_uniform_run1/'
    'gmm_models/vade_6clusters_merged_control_degron_rep1_relu_sep_pi_uniform_run1.pkl')
OUT_DIR = os.path.join(REPO_ROOT, 'scripts/gmm_metric_validation/results/trainable_prior_test')

N_CENTROID = 6
EPOCHS = 300
CHECK_EVERY = 20
SEED = 48


class CheckClusters(tf.keras.callbacks.Callback):
    """Records the RAW (un-gathered) argmax cluster distribution every CHECK_EVERY
    epochs -- the same diagnostic used for the em/gmm_fixed collapse epoch tables in
    run_report.md, so this is directly comparable to those."""
    def __init__(self, data, tag, vade):
        super().__init__()
        self.data = data
        self.tag = tag
        self.vade = vade  # not self.model -- Keras only sets that once fit() starts,
                           # but _record(0) is called before fit() for the init baseline
        self.history = []

    def _record(self, epoch):
        z_mean, _, _ = self.vade.encoder.predict(self.data, verbose=0)
        prob = self.vade.gmm(z_mean).numpy()
        cluster = np.argmax(prob, axis=1)
        uniq, counts = np.unique(cluster, return_counts=True)
        full_counts = [0] * N_CENTROID
        for c, n in zip(uniq, counts):
            full_counts[c] = int(n)
        max_frac = max(full_counts) / len(cluster)
        entry = {'epoch': epoch, 'n_populated': len(uniq), 'counts': full_counts,
                  'max_frac': float(max_frac)}
        self.history.append(entry)
        print(f"[{self.tag}] epoch {epoch}: {len(uniq)}/{N_CENTROID} populated, "
              f"max_frac={max_frac:.3f}, counts={full_counts}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % CHECK_EVERY == 0:
            self._record(epoch + 1)


def run(tag, trainable_prior):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    data = np.load(DATA_PATH)
    d_input = data.shape[1]

    vade = VADE(d_input, N_CENTROID, trainable_theta=False, trainable_prior=trainable_prior)
    vade(np.zeros((10, d_input)))

    out_path = os.path.join(OUT_DIR, tag) + '/'
    os.makedirs(out_path, exist_ok=True)
    vade.load_pretrained_weights(PRETRAIN_PATH, data, f'trainable_prior_test_{tag}', out_path,
                                 prior_gmm_path=GMM_PRIOR_PATH)

    checker = CheckClusters(data, tag, vade)
    checker._record(0)  # baseline right after init, before any training step

    adam = tf.keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-4)
    vade.compile(optimizer=adam)
    vade.fit(data, shuffle=True, batch_size=256, epochs=EPOCHS, callbacks=[checker], verbose=2)

    return checker.history


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for tag, trainable_prior in [('frozen', False), ('trainable', True)]:
        print(f"===== {tag} (trainable_prior={trainable_prior}) =====", flush=True)
        results[tag] = run(tag, trainable_prior)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {os.path.join(OUT_DIR, 'results.json')}")
