"""User-confirmed task4 ablations; combined method stays separately accessible."""
from inchworm_variant import VariantConfig
from inchworm_core import Config

DEFAULT_ABLATIONS=('guard_only','guard_temperature','guard_gap')
VARIANTS=(*DEFAULT_ABLATIONS,'combined')
BENCHMARK_VERSIONS=(*VARIANTS,'original','backbone')


def variant_config(version):
    modes={
        'guard_only':('entropy','topk'),
        'guard_temperature':('entropy','temperature'),
        'guard_gap':('mixed','topk'),
        'combined':('mixed','temperature'),
    }
    if version not in modes: raise ValueError('Unknown variant version')
    position,proposal=modes[version]
    config=VariantConfig(position_mode=position,proposal_mode=proposal)
    config.validate()
    return config


def ablation_flags(version):
    if version in ['original','backbone']:
        return {'original_path_guard':False,'adaptive_temperature':False,'mixed_high_H_high_G':False}
    config=variant_config(version)
    return {'original_path_guard':True,'adaptive_temperature':config.proposal_mode=='temperature',
            'mixed_high_H_high_G':config.position_mode=='mixed'}


def benchmark_config(version):
    return Config() if version in ['original','backbone'] else variant_config(version)


def controller_method(version):
    if version not in BENCHMARK_VERSIONS: raise ValueError('Unknown benchmark version')
    if version=='backbone': return 'backbone'
    return 'inchworm' if version=='original' else 'inchworm_variant'
