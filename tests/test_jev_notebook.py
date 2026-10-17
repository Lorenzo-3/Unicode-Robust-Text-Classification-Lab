"""Check the delivered notebooks' helpers offline, without executing their run cells."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nlp-pruned-test-matplotlib')
ROOT = Path(__file__).resolve().parents[1]


def definitions(name):
    namespace = {}
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        for cell in notebook['cells']:
            if 'definitions' in cell.get('metadata', {}).get('tags', []):
                exec(compile(''.join(cell['source']), name, 'exec'), namespace)
    finally:
        os.chdir(previous)
    return namespace


N = definitions('07_jev_robustness.ipynb')
CORE = definitions('06_results_and_statistics.ipynb')
np, pd = N['np'], N['pd']
LABELS = {'trec': {0: 'a', 1: 'b', 2: 'c'}}


def payload(text='Cafe\u0301\u200b \U0001d400'):
    return {'model': N['MODEL'], 'state': {'text': text}, 'questions': {
        'classification': {'type': 'choice', 'instructions': 'Classify `text`.',
                           'criteria': {'a': 'First category', 'b': 'Second category'}}}}


def response(choice='a'):
    return {'model': N['MODEL'], 'answers': {'classification': {
        'type': 'choice', 'choice': choice, 'probabilities': {'a': .75, 'b': .25}, 'confidence': .5}}}


class FakeResponse:
    status_code = 200

    def __init__(self, body=None):
        self.body = body or response()

    def json(self):
        return self.body


def fixture():
    rows = []
    for attack in N['CONDITIONS']:
        for i, gt_label in enumerate([0, 0, 1, 1]):
            prediction = gt_label if attack in ['clean', 'invisible'] or i != 0 else 1 - gt_label
            rows.append(dict(dataset='trec', model='word_tfidf', model_seed=42,
                replicate='42', example_id=f'e{i}', attack=attack,
                strength=0.0 if attack == 'clean' else .20,
                attack_seed=-1 if attack == 'clean' else 13, defense='none',
                gt_label=gt_label, prediction=prediction, correct=prediction == gt_label,
                changed_count=int(attack != 'clean')))
    return pd.DataFrame(rows)


class JevCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.dict(N, JEV_DIR=self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.environment = patch.dict(os.environ, {'JEV_OFFLINE': '0'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def store(self, value=None, body=None):
        value = value or payload()
        key = N['digest'](value)
        path = self.root / 'responses' / f'{key}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(N['canonical_bytes']({'request_hash': key, 'payload': value, 'response': body or response()}))
        return path

    def test_cached_response_never_calls_api(self):
        self.store()
        with patch.object(N['requests'], 'post', side_effect=AssertionError('Network called')):
            answer = N['ask'](payload(), None)
        self.assertEqual(answer['choice'], 'a')

    def test_missing_cache_requires_explicit_opt_in_and_key(self):
        for live, key in [(False, 'secret'), (True, None)]:
            with patch.object(N['requests'], 'post', side_effect=AssertionError('Network called')):
                with self.assertRaisesRegex(ValueError, 'Missing response'):
                    N['ask'](payload(), key, live)

    def test_forced_offline_overrides_live_flag(self):
        with patch.dict(os.environ, {'JEV_OFFLINE': '1'}), patch.object(N['requests'], 'post') as post:
            with self.assertRaisesRegex(ValueError, 'Missing response'):
                N['ask'](payload(), 'secret', True)
            post.assert_not_called()

    def test_live_request_preserves_unicode_and_caches_immediately(self):
        with patch.object(N['requests'], 'post', return_value=FakeResponse()) as post:
            N['ask'](payload(), 'test-secret', True)
        kwargs = post.call_args.kwargs
        self.assertIn(payload()['state']['text'].encode(), kwargs['data'])
        self.assertEqual(json.loads(kwargs['data']), payload())
        self.assertEqual(set(json.loads(kwargs['data'])['state']), {'text'})
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(kwargs['timeout'], (30, 90))
        path = self.root / 'responses' / f"{N['digest'](payload())}.json"
        self.assertNotIn('test-secret', path.read_text())
        with patch.object(N['requests'], 'post', side_effect=AssertionError('Network called')):
            N['ask'](payload(), None)

    def test_model_question_or_text_changes_invalidate_cache(self):
        self.store()
        for field in ['model', 'question', 'text']:
            changed = copy.deepcopy(payload())
            if field == 'model':
                changed['model'] += '-new'
            elif field == 'text':
                changed['state']['text'] += '\u200b'
            else:
                changed['questions']['classification']['criteria']['a'] = 'Different'
            with self.assertRaisesRegex(ValueError, 'Missing response'):
                N['ask'](changed, None)

    def test_cache_payload_mismatch_is_rejected(self):
        path = self.store()
        record = json.loads(path.read_text())
        record['payload']['state']['text'] = 'Wrong input'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Cache/request mismatch'):
            N['ask'](payload(), None)

    def test_invalid_responses_are_not_cached(self):
        invalid = [response('unknown'), response(), response(), response(), response()]
        invalid[1]['model'] = 'wrong'
        invalid[2]['answers']['classification']['probabilities'] = {'a': .2, 'b': .2}
        invalid[3]['answers']['classification']['probabilities'] = {'a': float('nan'), 'b': .25}
        invalid[4]['answers']['classification']['probabilities'] = {'a': 1.1, 'b': -.1}
        for body in invalid:
            with patch.object(N['requests'], 'post', return_value=FakeResponse(body)):
                with self.assertRaises(ValueError):
                    N['ask'](payload(), 'secret', True)
        self.assertFalse(list(self.root.rglob('*.json')))

    def test_returned_choice_and_rounded_probabilities_are_preserved(self):
        body = response('b')
        body['answers']['classification']['probabilities']['b'] = .24
        self.store(body=body)
        answer = N['ask'](payload(), None)
        self.assertEqual(answer['choice'], 'b')
        self.assertEqual(answer['probabilities']['b'], .24)

    def test_http_failure_stops_without_caching(self):
        bad = FakeResponse()
        bad.status_code = 429
        with patch.object(N['requests'], 'post', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'HTTP 429'):
                N['ask'](payload(), 'secret', True)
        self.assertFalse(list(self.root.rglob('*.json')))

    def test_environment_key_precedes_private_file(self):
        (self.root / '.env.local').write_text('TYPESAFE_API_KEY="file-key"\n')
        with patch.dict(N, ROOT=self.root), patch.dict(os.environ, {'TYPESAFE_API_KEY': 'env-key'}):
            self.assertEqual(N['read_api_key'](), 'env-key')
        with patch.dict(N, ROOT=self.root), patch.dict(os.environ, {'TYPESAFE_API_KEY': ''}):
            self.assertEqual(N['read_api_key'](), 'file-key')

    def test_evaluate_deduplicates_and_keeps_rows_matched(self):
        conditions = pd.DataFrame([
            dict(dataset='trec', example_id='e1', attack='clean', strength=0., attack_seed=-1,
                 text='Same text', label=0, label_name='a', changed_count=0),
            dict(dataset='trec', example_id='e1', attack='invisible', strength=.2, attack_seed=13,
                 text='Same text', label=0, label_name='a', changed_count=0),
        ])
        question = payload()['questions']['classification']
        with patch.dict(N, read_api_key=lambda: 'secret'), patch.object(N['requests'], 'post', return_value=FakeResponse()) as post:
            result = N['evaluate'](conditions, {'trec': question}, {'trec': {0: 'a', 1: 'b'}}, True)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result.prediction.tolist(), [0, 0])
        self.assertEqual(result.gt_probability.tolist(), [.75, .75])
        self.assertEqual(result.attack.tolist(), ['clean', 'invisible'])

    def test_cache_is_reused_across_strengths_and_choice_is_the_prediction(self):
        conditions = pd.DataFrame([
            dict(dataset='trec', example_id='e1', attack='mixed', strength=s, attack_seed=13,
                 text=payload()['state']['text'], label=0, label_name='a', changed_count=1)
            for s in [.05, .20, .75]])
        self.store(body=response('b'))
        with patch.object(N['requests'], 'post', side_effect=AssertionError('Network called')):
            result = N['evaluate'](conditions, {'trec': payload()['questions']['classification']},
                                   {'trec': {0: 'a', 1: 'b'}}, False)
        self.assertEqual(result.prediction.tolist(), [1, 1, 1])
        self.assertEqual(result.strength.tolist(), [.05, .20, .75])
        self.assertEqual(result.request_hash.nunique(), 1)


class MetricTests(unittest.TestCase):
    def test_scores_use_all_classes(self):
        gt_label, prediction = np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1])
        score = N['scores'](gt_label, prediction, 3)
        self.assertEqual(score[0], .75)
        self.assertAlmostEqual(score[1], N['f1_score'](gt_label, prediction, labels=[0, 1, 2], average='macro', zero_division=0))

    def test_paired_rows_and_equal_family_mean(self):
        frame = fixture().sample(frac=1, random_state=12)
        per_seed, summary, aggregate_seed, aggregate = N['metric_tables'](frame, LABELS)
        self.assertAlmostEqual(aggregate.iloc[0].attack_success_rate_mean, 3 * .25 / 4)
        self.assertEqual(per_seed[per_seed.attack.eq('invisible')].iloc[0].macro_f1_drop, 0)
        second = fixture().assign(replicate='73')
        _, _, _, doubled = N['metric_tables'](pd.concat([frame, second]), LABELS)
        self.assertAlmostEqual(doubled.iloc[0].macro_f1_mean, aggregate.iloc[0].macro_f1_mean)

    def test_missing_duplicate_and_mismatched_pairs_are_rejected(self):
        frame = fixture()
        wrong = frame.copy()
        wrong.loc[wrong.attack.eq('mixed') & wrong.example_id.eq('e0'), 'gt_label'] = 1
        for changed in [frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), wrong]:
            with self.assertRaises(ValueError):
                N['metric_tables'](changed, LABELS)

    def test_undefined_asr_stays_undefined(self):
        frame = fixture()
        frame['prediction'] = 1 - frame.gt_label
        frame['correct'] = False
        per_seed, summary, _, aggregate = N['metric_tables'](frame, LABELS)
        self.assertTrue(per_seed.attack_success_rate.isna().all())
        self.assertTrue(summary.attack_success_rate_mean.isna().all())
        self.assertTrue(aggregate.attack_success_rate_mean.isna().all())
        core = pd.DataFrame(CORE['condition_metric_rows'](frame))
        self.assertTrue(core[core.attack.ne('clean')].attack_success_rate.isna().all())

    def test_core_requires_clean_counterparts(self):
        frame = fixture()
        frame = frame[~(frame.attack.eq('clean') & frame.example_id.eq('e0'))]
        with self.assertRaisesRegex(ValueError, 'Missing clean'):
            CORE['condition_metric_rows'](frame)

    def test_baseline_alignment_checks_identity_and_completeness(self):
        frame = fixture()
        conditions = frame.rename(columns={'gt_label': 'label'})
        settings = {'COUNTS': {'trec': 4}, 'BASELINES': ['word_tfidf']}
        with patch.dict(N, settings), patch.object(pd, 'read_parquet', return_value=frame.sample(frac=1, random_state=2)):
            self.assertEqual(len(N['load_baselines'](conditions)), len(frame))
        for bad in [frame.iloc[1:], frame.assign(model_seed=99)]:
            with patch.dict(N, settings), patch.object(pd, 'read_parquet', return_value=bad):
                with self.assertRaises(ValueError):
                    N['load_baselines'](conditions)

    def test_core_summary_preserves_undefined_recovery(self):
        #Run the actual summary cell against synthetic per-condition metrics.
        rows = pd.DataFrame(CORE['condition_metric_rows'](fixture()))
        rows['accuracy'] = 1.0
        rows['macro_f1'] = 1.0
        defended = rows.assign(defense='nfkc')
        combined = pd.concat([rows, defended], ignore_index=True)
        notebook = json.loads((ROOT / 'notebooks/06_results_and_statistics.ipynb').read_text())
        cell = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and "seed_metrics = pd.DataFrame(seed_rows)" in ''.join(c['source']))
        cell = cell[cell.index('seed_metrics = pd.DataFrame(seed_rows)'):]
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(CORE, seed_rows=combined.to_dict('records'), PATHS={'tables': Path(directory)})
            exec(compile(cell, '<core summary>', 'exec'), namespace)
        self.assertTrue(namespace['summary'].macro_f1_defense_recovery.isna().all())
        self.assertTrue(namespace['summary'].accuracy_defense_recovery.isna().all())


class StrengthMetricTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict(N, COUNTS={'trec': 4})
        self.settings.start()
        self.addCleanup(self.settings.stop)
        original = fixture().assign(model='jev', replicate='api')
        self.frame = pd.concat([original[original.attack.eq('clean')]] + [
            original[original.attack.ne('clean')].assign(strength=s) for s in N['STRENGTHS']], ignore_index=True)

    def test_all_strengths_keep_the_original_metrics_and_equal_family_weights(self):
        conditions, summary = N['strength_metric_tables'](self.frame.sample(frac=1, random_state=4), LABELS)
        self.assertEqual(len(conditions), 21)
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary.n.tolist(), [4] * 6)
        self.assertEqual(conditions[conditions.attack.eq('clean')].strength.tolist(), [0.])
        original = N['metric_tables'](fixture().assign(model='jev', replicate='api'), LABELS)[0]
        retained = conditions[conditions.strength.isin([0., .2])].drop(columns='strength')
        pd.testing.assert_frame_equal(original.sort_values('attack').reset_index(drop=True),
                                      retained.sort_values('attack').reset_index(drop=True))
        self.assertTrue(np.allclose(summary.loc[summary.strength.gt(0), 'attack_success_rate'], .1875))
        #Only change 75% mixed: other strengths must retain their own predictions.
        changed = self.frame.copy()
        mask = changed.strength.eq(.75) & changed.attack.eq('mixed')
        changed.loc[mask, 'prediction'] = 1 - changed.loc[mask, 'gt_label']
        _, result = N['strength_metric_tables'](changed, LABELS)
        self.assertAlmostEqual(result.loc[result.strength.eq(.75), 'attack_success_rate'].iloc[0], .375)
        self.assertAlmostEqual(result.loc[result.strength.eq(.2), 'attack_success_rate'].iloc[0], .1875)

    def test_missing_whole_strength_family_example_or_wrong_pair_fails(self):
        wrong_label = self.frame.copy()
        wrong_label.loc[wrong_label.strength.eq(.75) & wrong_label.attack.eq('mixed'), 'gt_label'] = 2
        wrong_id = self.frame.copy()
        wrong_id.loc[0, 'example_id'] = 'not-the-clean-example'
        for frame in [self.frame[self.frame.strength.ne(.75)], self.frame[self.frame.attack.ne('mixed')],
                      self.frame[self.frame.example_id.ne('e0')], self.frame.iloc[1:],
                      pd.concat([self.frame, self.frame.iloc[:1]]), wrong_label, wrong_id,
                      self.frame.assign(attack_seed=99)]:
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                N['strength_metric_tables'](frame, LABELS)

    def test_undefined_asr_survives_family_averaging(self):
        frame = self.frame.assign(prediction=1 - self.frame.gt_label)
        conditions, summary = N['strength_metric_tables'](frame, LABELS)
        self.assertTrue(conditions.attack_success_rate.isna().all())
        self.assertTrue(summary.attack_success_rate.isna().all())


if __name__ == '__main__':
    unittest.main()
