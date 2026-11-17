"""Read existing GenEval receipts. No models, GPU, network or writes required."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path


MODES = {
    'guard_only': ('entropy', 'topk'),
    'guard_temperature': ('entropy', 'temperature'),
    'guard_gap': ('mixed', 'topk'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(x, y):
    return math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-10)


def audit_event(e, version, totals):
    pos_mode, prop_mode = MODES[version]
    require((e['position_mode'], e['proposal_mode']) == MODES[version], 'wrong runtime modes')
    s, f = e['s'], e['f']
    h = e['original_entropies']
    sites = e['positions']
    top_h = sorted(range(s+1, f+1), key=lambda i: (-h[i-s-1], i))[:4]
    require(len(h) == f-s and len(sites) == len(set(sites)) == min(4, f-s), 'window/sites')
    require(all(s < i <= f for i in sites), 'site outside window')
    require(sites[:2] == top_h[:2] if pos_mode == 'mixed' else sites == top_h, 'H ranking')
    added = set(sites) - set(top_h)
    totals['window_le_4'] += int(f-s <= 4)
    totals['different_site_set_vs_top4H'] += bool(added)
    props = e['proposals']
    require([p['position'] for p in props] == sites, 'proposal sites')
    for p in props:
        i = p['position']-s-1
        draws = p['sampled_tokens']
        require(len(draws) == 2 and p['original_token'] == e['original_tokens'][i], 'draws')
        require(close(p['base_entropy'], h[i]) and p['mode'] == prop_mode, 'proposal distribution')
        if prop_mode == 'temperature':
            require(close(p['temperature'], .5+.5*math.exp(-h[i]/2)), 'temperature')
            totals['duplicate_draw_pairs'] += draws[0] == draws[1]
            totals['original_token_draws'] += draws.count(p['original_token'])
            totals['draws'] += len(draws)
        else:
            require(p['temperature'] is None and len(set(draws)) == 2 and p['original_token'] not in draws, 'topk')
        require(p['gap'] is not None and p['gap'] >= 0 if pos_mode == 'mixed' else p['gap'] is None, 'gap')
    candidates = e['candidates']
    require([(c['position'], c['alternative']) for c in candidates] ==
            [(p['position'], t) for p in props for t in p['sampled_tokens']], 'candidate mapping')
    ranked = sorted(candidates, key=lambda c: (c['mean_entropy'], c['id']))[:2]
    finalists = e['survivors']
    require([c['id'] for c in ranked] == [v['id'] for v in finalists], 'stage-one ranking')
    terminal = f == 576
    def score(mean, look):
        require((look is None) == terminal, 'terminal convention')
        return mean if terminal else .5*(mean+look)
    for c,v in zip(ranked, finalists):
        require(close(c['mean_entropy'],v['mean_entropy']) and
                close(v['score'],score(v['mean_entropy'],v['lookahead_entropy'])), 'finalist score')
    orig = score(sum(h)/len(h), e['original_lookahead_entropy'])
    require(close(orig, e['original_score']), 'original score')
    best = min(finalists,key=lambda v:(v['score'],v['id']))
    require(e['improvement_margin'] == 0 and close(best['score'],e['best_candidate_score']), 'best score')
    accepted = best['score'] < e['original_score']
    require(e['accepted'] == accepted and e['selected_id'] == (best['id'] if accepted else 'original'), 'guard decision')
    require(close(e['selected_score'], best['score'] if accepted else orig), 'selected score')
    if not accepted:
        require(e['selected_tokens'] == e['original_tokens'], 'rejected path changed')
    totals['checks'] += 1
    totals['accepted'] += accepted
    totals['kept'] += not accepted
    totals['changed_windows'] += e['selected_tokens'] != e['original_tokens']
    if accepted:
        winning = next(c for c in ranked if c['id'] == best['id'])
        totals['accepted_from_site_outside_top4H'] += winning['position'] in added


def inspect(root, version):
    method = 'janus1b_'+version
    directory = root/method
    totals = Counter()
    hashes = {}
    sources = set()
    for i in range(553):
        for j in range(4):
            path = directory/f'{i:05}'/'records'/f'{j:04}.json'
            try:
                r = json.loads(path.read_bytes())
                spec = r['specification']
                require(spec['method'] == method and spec['variant'] == version and
                        r['seed'] == 42+4*i+j, 'identity/seed')
                sources.add(json.dumps(spec['source_sha256'], sort_keys=True))
                ev = [e for e in r['events'] if e['kind'] == 'check']
                accepted = sum(e['accepted'] for e in ev)
                require((len(ev), accepted, len(ev)-accepted) ==
                        (r['checks'],r['accepted_edits'],r['kept_original']), 'image counters')
                for e in ev:
                    audit_event(e,version,totals)
                totals['images_with_checks'] += bool(ev)
                totals['images_with_edits'] += bool(accepted)
                hashes[i,j] = r['image_sha256']
            except (ValueError, KeyError, OSError, IndexError, TypeError) as err:
                raise ValueError(f'{path}: {err}') from err
    require(len(sources) == 1, 'mixed generator sources within method')
    print(version, 'images',len(hashes), dict(sorted(totals.items())), flush=True)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cvpr',type=Path)
    a = parser.parse_args()
    roots = {v:a.cvpr/('janus_rescore_20261113' if v=='guard_only' else
                         'janus1b_ablation_recovery_20261115')/'merged' for v in MODES}
    hashes = {v:inspect(root,v) for v,root in roots.items()}
    grades = {}
    for v, root in roots.items():
        method = 'janus1b_'+v
        folder = root/'evaluation'
        results = {}
        category = {}
        for line in (folder/(method+'.jsonl')).read_text().splitlines():
            row = json.loads(line)
            path = Path(row['filename'])
            key = int(path.parents[1].name), int(path.stem)
            require(path.parent.name == 'samples' and path.parents[2].name == method and
                    key not in results and isinstance(row['correct'],bool), 'unexpected evaluated image')
            results[key] = row['correct']
            category.setdefault(row['tag'], []).append(row['correct'])
        require(results.keys() == hashes[v].keys() and len(category) == 6, 'scored image coverage')
        summary = json.loads((folder/(method+'.summary.json')).read_bytes())
        scores = {k:sum(vals)/len(vals) for k,vals in category.items()}
        require(all(close(score,summary['task_scores'][k]) for k,score in scores.items())
                and close(sum(scores.values())/6,summary['overall']), 'summary mismatch')
        grades[v] = results
    for a,b in combinations(hashes,2):
        same = sum(h == hashes[b][key] for key,h in hashes[a].items())
        wins = sum(grades[b][key] and not ok for key,ok in grades[a].items())
        losses = sum(ok and not grades[b][key] for key,ok in grades[a].items())
        print(a,'->',b,'same_stored_png_hash',same,'of',len(hashes[a]),
              'incorrect_to_correct',wins,'correct_to_incorrect',losses)


if __name__ == '__main__':
    main()
