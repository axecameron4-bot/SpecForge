"""Verify the research runner's supervision and all-layer conditioning."""
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel, build_target_layer_ids
from specforge.research.qwen35_dflash import encode
from specforge.research.benchmark_qwen35_dflash import FullBlockBenchmarkModel, answer_number


class TestQwen35DFlashResearch(unittest.TestCase):
    def test_all_target_layers_receive_projection_gradients(self):
        torch.manual_seed(4)
        config = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             vocab_size=64, layer_types=['full_attention']*2)
        config.num_target_layers = 6
        config.block_size = 4
        config.dflash_config = {'target_layer_ids': list(range(6)), 'mask_token_id': 63}
        config._attn_implementation = 'sdpa'
        draft = DFlashDraftModel(config)
        head = nn.Linear(16, 64, bias=False).requires_grad_(False)
        embedding = nn.Embedding(64, 16).requires_grad_(False)
        wrapper = OnlineDFlashModel(draft, head, embedding, mask_token_id=63,
                                   block_size=4, attention_backend='sdpa', num_anchors=2,
                                   objective_chunk_blocks=1, teacher_metrics=False)
        tokens = torch.randint(0, 60, (1, 32))
        features = torch.randn(1, 32, 6*16)
        mask = torch.zeros(1, 32)
        mask[:, 8:] = 1
        loss, _, _ = wrapper(tokens, features, mask, collect_detailed_metrics=False)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for layer in range(6):
            grad = draft.fc.weight.grad[:, layer*16:(layer+1)*16]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(embedding.weight.grad)
        self.assertEqual(build_target_layer_ids(24, 5), [1, 6, 11, 16, 21])

    def test_benchmark_samples_only_complete_supervised_blocks(self):
        sampler = object.__new__(FullBlockBenchmarkModel)
        nn.Module.__init__(sampler)
        sampler.block_size = 16
        sampler.num_anchors = 64
        mask = torch.zeros(1, 64)
        mask[:, 20:] = 1
        anchors, keep = sampler._sample_anchor_positions(64, mask, torch.device('cpu'))
        self.assertTrue(keep.all())
        self.assertTrue((anchors >= 20).all())
        self.assertTrue((anchors + 16 <= 64).all())
        self.assertEqual(anchors.numel(), 29)
        self.assertEqual(answer_number('Reason 2 + 3 = 5. Answer: #### 5'), '5')
        self.assertEqual(answer_number('Final: \u005c\u005cboxed{1,234}'), '1234')

    @unittest.skipUnless(Path('cache/models/Qwen3.5-0.8B/tokenizer.json').exists(),
                         'Requires the locally downloaded Qwen3.5 tokenizer')
    def test_real_tokenizer_assistant_boundary_and_prompt_truncation(self):
        tokenizer = AutoTokenizer.from_pretrained('cache/models/Qwen3.5-0.8B', local_files_only=True)
        row = {'conversations': [{'role': 'user', 'content': 'Explain addition.'},
                                 {'role': 'assistant', 'content': 'Adding two numbers means combining their quantities. '*20}]}
        encoded = encode(row, tokenizer, 128, device='cpu')
        self.assertIsNotNone(encoded)
        tokens, mask = encoded
        self.assertEqual(tokens.shape, mask.shape)
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertGreater(mask.sum().item(), 16)
        row['conversations'][0]['content'] = 'A long question. '*1000
        self.assertIsNone(encode(row, tokenizer, 32, device='cpu'))


if __name__ == '__main__':
    unittest.main()
