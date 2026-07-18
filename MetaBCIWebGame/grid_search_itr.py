"""Focused grid search around known good thresholds."""
import sys, os, glob, numpy as np
from math import log2
sys.path.insert(0, os.path.join(os.path.dirname('.'), '..'))
_M = os.path.join(os.path.dirname(os.path.abspath('.')), '..', 'metabci')
if os.path.isdir(os.path.join(_M, 'brainda')): sys.path.insert(0, os.path.abspath(os.path.join(_M, '..')))
from config import GW_MODEL_PATHS, OCCIPITAL_INDICES, BASE_DIR
from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder

X, y = [], []
for label in range(4):
    for f in glob.glob(os.path.join(BASE_DIR, 'data_self_test', str(label+1), '*offset000.npy')):
        X.append(np.load(f)[OCCIPITAL_INDICES, :]); y.append(label)

margins = [round(x,2) for x in np.arange(0.20, 0.45, 0.05)]
max_scores = [round(x,2) for x in np.arange(0.45, 0.85, 0.05)]
results = []
total = len(margins) * len(max_scores)
done = 0

for margin in margins:
    for max_s in max_scores:
        dec = GrowingWindowDecoder(model_paths=GW_MODEL_PATHS, margin_th=margin, max_th=max_s)
        correct = 0; times = []
        for data, label in zip(X, y):
            dec.reset(); dec.reset_normaliser()
            d = None
            for i in range(data.shape[1]):
                d, conf, t = dec.feed(data[:, i])
                if d is not None: break
            if d == label: correct += 1
            if t > 0: times.append(t)
        acc = correct / len(X)
        done += 1
        if acc < 0.95: continue
        avg_t = np.mean(times) if times else 2.0
        early = sum(1 for t in times if t<2.0)/len(times) if times else 0
        P=correct/len(X); T=np.mean(times)
        B=log2(4)+P*log2(P)+(1-P)*log2((1-P)/3) if P>0.25 else 0
        itr=B*60/T
        results.append({'margin':margin,'max':max_s,'acc':acc*100,'t_ms':avg_t*1000,'early':early*100,'itr':itr})
        print(f'{done}/{total} margin={margin:.2f} max={max_s:.2f} acc={acc*100:.1f}% t={avg_t*1000:.0f}ms', flush=True)

results.sort(key=lambda r:-r['itr'])
header = '{:>7} {:>7} {:>7} {:>7} {:>7} {:>7}'.format('margin','max','acc%','t_ms','early%','ITR')
print()
print(header)
print('-' * 49)
for r in results:
    s = '{:>7.2f} {:>7.2f} {:>7.2f} {:>7.0f} {:>7.1f} {:>7.1f}'.format(
        r['margin'], r['max'], r['acc'], r['t_ms'], r['early'], r['itr'])
    print(s)
print(f'\n共 {len(results)} 组满足 acc >= 95%')
