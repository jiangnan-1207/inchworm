"""User-confirmed task4 ablations; combined method stays separately accessible."""
from inchworm_variant import VariantConfig

DEFAULT_ABLATIONS=('guard_only','guard_temperature','guard_gap')
VARIANTS=(*DEFAULT_ABLATIONS,'combined')


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
    config=variant_config(version)
    return {'original_path_guard':True,'adaptive_temperature':config.proposal_mode=='temperature',
            'mixed_high_H_high_G':config.position_mode=='mixed'}
