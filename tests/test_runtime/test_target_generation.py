"""Target generation must replace dataset answers before feature capture."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from specforge.config import Config
from specforge.data.target_generation_prompts import prepare_generation_prompts
from specforge.inference.adapters.target_generation import SGLangTargetGenerationAdapter
from tests.test_config.test_server_only_online import _online_payload
from tests.test_runtime.test_server_capture import (
    _FakeMooncakeStore, _StubCaptureServer, _dflash_contract, _mk, _task,
)


class TargetGenerationTest(unittest.TestCase):
    def test_generation_then_real_capture_transport_and_masks(self):
        backend = _FakeMooncakeStore()
        sink = _StubCaptureServer(backend)
        calls = []

        def server(url, json_body, timeout):
            calls.append(json_body)
            if 'spec_capture' not in json_body:
                return [{'output_ids': [20, 21, 22], 'meta_info': {'finish_reason': {'type': 'stop'}}},
                        {'output_ids': [30, 31], 'meta_info': {}}]
            return sink(url, json_body, timeout)

        _, _, store, capture = _mk(algorithm='dflash', server=server, backend=backend, teacher_metrics=False)
        tasks = [replace(_task(i, 3), payload={'input_ids': [1, 2, 3], 'loss_mask': [0, 0, 0]},
                         max_length=8, metadata={'target_generation_prompt': True}) for i in range(2)]
        adapter = SGLangTargetGenerationAdapter(capture, max_new_tokens=10)
        refs = adapter.produce_refs(tasks, capture=_dflash_contract())
        self.assertEqual(calls[0]['input_ids'], [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(calls[0]['sampling_params'][0]['max_new_tokens'], 5)
        self.assertEqual(calls[1]['sampling_params']['max_new_tokens'], 0)
        self.assertEqual(calls[1]['input_ids'], [[1, 2, 3, 20, 21, 22], [1, 2, 3, 30, 31]])
        for ref, expected in zip(refs, [[20, 21, 22], [30, 31]]):
            tensors, handle = store.get(ref)
            self.assertEqual(tensors['input_ids'].flatten().tolist(), [1, 2, 3]+expected)
            self.assertEqual(tensors['loss_mask'].flatten().tolist(), [0, 0, 0]+[1]*len(expected))
            self.assertEqual(ref.num_tokens, 3+len(expected))
            self.assertEqual(tensors['hidden_states'].shape[1], ref.num_tokens)
            self.assertTrue(ref.metadata['target_generated'])
            store.release(handle)
        self.assertEqual(tasks[0].payload['input_ids'], [1, 2, 3])

    def test_greedy_cache_reuses_answers_but_recomputes_capture(self):
        backend = _FakeMooncakeStore()
        sink = _StubCaptureServer(backend)
        calls = []

        def server(url, json_body, timeout):
            calls.append(json_body)
            if 'spec_capture' not in json_body:
                return [{'output_ids': [20, 21, 22], 'meta_info': {}}]
            return sink(url, json_body, timeout)

        _, _, store, capture = _mk(algorithm='dflash', server=server, backend=backend, teacher_metrics=False)
        task = replace(_task(0, 3), payload={'input_ids': [1, 2, 3], 'loss_mask': [0]*3},
                       max_length=16, metadata={'target_generation_prompt': True})
        with tempfile.TemporaryDirectory() as directory:
            adapter = SGLangTargetGenerationAdapter(capture, response_cache_dir=directory,
                                                    cache_namespace='pinned-model-sha')
            first = adapter.produce_refs([task], capture=_dflash_contract())[0]
            second = adapter.produce_refs([replace(task, task_id='epoch-2')], capture=_dflash_contract())[0]
            self.assertFalse(first.metadata['response_cache_hit'])
            self.assertTrue(second.metadata['response_cache_hit'])
        self.assertEqual(len(calls), 3)
        self.assertNotIn('spec_capture', calls[0])
        self.assertIn('spec_capture', calls[1])
        self.assertIn('spec_capture', calls[2])
        self.assertEqual(calls[1]['input_ids'], calls[2]['input_ids'])
        with self.assertRaisesRegex(ValueError, 'greedy'):
            SGLangTargetGenerationAdapter(capture, temperature=0.5, response_cache_dir='cache',
                                          cache_namespace='pinned')

    def test_prompt_preparation_never_reads_dataset_answer(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return [5, 6, 7]
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.jsonl'
            path.write_text(json.dumps({'id': 'abc', 'conversations': [
                {'role': 'user', 'content': 'Question'},
                {'role': 'assistant', 'content': {'poison': 'must never be tokenized'}},
            ]})+'\n')
            rows = prepare_generation_prompts(path, tokenizer, max_length=16)
        self.assertEqual(tokenizer.messages, [{'role': 'user', 'content': 'Question'}])
        self.assertTrue(tokenizer.kwargs['add_generation_prompt'])
        self.assertEqual(rows[0]['payload']['loss_mask'], [0, 0, 0])
        self.assertEqual(rows[0]['max_length'], 16)
        self.assertTrue(rows[0]['metadata']['target_generation_prompt'])

    def test_rejects_old_answer_supervision_and_missing_exact_tokens(self):
        _, _, _, capture = _mk(algorithm='dflash', teacher_metrics=False)
        adapter = SGLangTargetGenerationAdapter(capture)
        with self.assertRaisesRegex(ValueError, 'prompt-only'):
            adapter.produce_refs([_task(0, 3)], capture=_dflash_contract())
        task = replace(_task(0, 3), payload={'input_ids': [1, 2, 3], 'loss_mask': [0]*3},
                       metadata={'target_generation_prompt': True}, max_length=16)
        capture.post_fn = lambda *a, **k: [{'text': 'Do not retokenize this'}]
        with self.assertRaisesRegex(ValueError, 'exact output_ids'):
            adapter.produce_refs([task], capture=_dflash_contract())
        capture.post_fn = lambda *a, **k: []
        with self.assertRaisesRegex(ValueError, 'response count'):
            adapter.produce_refs([task], capture=_dflash_contract())

    def test_config_is_explicit_and_rejects_unsupported_paths(self):
        payload = _online_payload()
        payload['training']['strategy'] = 'dflash'
        payload['data']['target_generation'] = {'max_new_tokens': 512}
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.data.target_generation.max_new_tokens, 512)
        payload['data']['train_data_path'] = ''
        payload['data']['prompts_path'] = 'old-tokens.jsonl'
        with self.assertRaisesRegex(ValueError, 'raw conversation'):
            Config.model_validate(payload)


if __name__ == '__main__':
    unittest.main()
