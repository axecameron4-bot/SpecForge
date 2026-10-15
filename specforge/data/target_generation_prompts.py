"""Prepare prompt-only tasks; never tokenize dataset assistant answers."""
from __future__ import annotations

from .prompt_builder import _iter_records


def prepare_generation_prompts(path, tokenizer, *, max_length, enable_thinking=False, max_prompts=None):
    prompts = []
    for line, row in _iter_records(str(path)):
        messages = row.get('conversations', row.get('messages'))
        if not isinstance(messages, list):
            raise ValueError(f'{path}:{line}: expected conversations or messages')
        prefix = []
        for message in messages:
            role = message.get('role', message.get('from'))
            role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
            if role == 'assistant':
                break
            if role not in ('system', 'user'):
                raise ValueError(f'{path}:{line}: unsupported prompt role {role!r}')
            content = message.get('content', message.get('value'))
            if not isinstance(content, str):
                raise ValueError(f'{path}:{line}: prompt content must be text')
            prefix.append({'role': role, 'content': content})
        if not prefix or prefix[-1]['role'] != 'user':
            raise ValueError(f'{path}:{line}: expected user prompt before first assistant')
        ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
        if not ids or len(ids) + 2 > max_length:
            continue
        prompts.append({'source_id': str(row.get('id', f'{path}:{line}')),
                        'max_length': max_length,
                        'payload': {'input_ids': list(ids), 'loss_mask': [0]*len(ids)},
                        'metadata': {'target_generation_prompt': True,
                                     'prompt_tokens': len(ids)}})
        if max_prompts and len(prompts) >= max_prompts:
            break
    return prompts
