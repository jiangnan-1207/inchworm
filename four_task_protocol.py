"""Full benchmark identities are separate from the 100-image profiling subset."""
VARIANT_METHODS={'janus1b':'janus1b_inchworm_variant','llamagen':'llamagen_inchworm_variant'}
from ablation_plan import VARIANTS


def variant_method(model,version='combined'):
    if model not in VARIANT_METHODS: raise ValueError('Task 4 supports Janus-Pro-1B and LlamaGen')
    if version not in VARIANTS: raise ValueError('Unknown variant version')
    return VARIANT_METHODS[model] if version=='combined' else f'{model}_{version}'


def full_shard_ids(index,shards):
    if not 0<=index<shards<=553: raise ValueError('Invalid full-benchmark shard')
    return list(range(index,553,shards))
