"""Full benchmark identities are separate from the 100-image profiling subset."""
VARIANT_METHODS={'janus1b':'janus1b_inchworm_variant','llamagen':'llamagen_inchworm_variant'}


def variant_method(model):
    if model not in VARIANT_METHODS: raise ValueError('Task 4 supports Janus-Pro-1B and LlamaGen')
    return VARIANT_METHODS[model]


def full_shard_ids(index,shards):
    if not 0<=index<shards<=553: raise ValueError('Invalid full-benchmark shard')
    return list(range(index,553,shards))
