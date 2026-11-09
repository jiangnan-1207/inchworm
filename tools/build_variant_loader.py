"""One-time development helper: derive a separate loader, leaving old entry intact.

Not called by experiment jobs. Rebuild only deliberately after reviewing the
original loader; exact-string guards prevent silently producing a partial port.
"""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=(ROOT/'run_requested_experiments.py').read_text(encoding='utf-8')
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_model')
    body='\n'.join(source.splitlines()[function.lineno-1:function.end_lineno])+'\n'
    replacements=[
        ('def load_model(name,root,data):','def load_model(name,root,data,variant_config=VariantConfig()):',1),
        ("if method=='inchworm':","if method in ['inchworm','inchworm_variant']:",2),
        ('generate(JanusBackend(model),State(prompt_inputs=inputs),576,Config())',
         'run_controller(JanusBackend(model),State(prompt_inputs=inputs),576,method,variant_config)',1),
        ('generate(backend,State(),1024,Config())',
         'run_controller(backend,State(),1024,method,variant_config)',1),
        ('return image,token_seconds,backtracks','return image,token_seconds,diagnostics(method,backtracks,locals().get("result"))',1),
        ('return image,seconds,backtracks','return image,seconds,diagnostics(method,backtracks,locals().get("result"))',1),
    ]
    for old,new,count in replacements:
        if body.count(old)!=count: raise ValueError(f'Original loader changed: {old}')
        body=body.replace(old,new)
    header='''"""Separate three-method loader; original experiment loader is untouched.

Model loading/sampling adapted from Janus and LlamaGen (see licenses/).
Derived by tools/build_variant_loader.py; review changes before rebuilding.
"""
import time
import sys
from inchworm_core import Config, generate
from inchworm_variant import VariantConfig, generate as generate_variant
from inchworm_variant_backend import VariantBackend


def run_controller(backend,state,length,method,config):
    if method=='inchworm_variant':
        return generate_variant(VariantBackend(backend,config.temperature,config.proposal_mode),state,length,config)
    if method=='inchworm':
        return generate(backend,state,length,Config())
    raise ValueError('Unknown controller')


def diagnostics(method,backtracks,result):
    if method=='backbone':
        return {'backtracks':0,'checks':0,'accepted_edits':0,'kept_original':0,'events':[]}
    return {'backtracks':backtracks,'checks':result.get('checks',backtracks),
        'accepted_edits':result.get('accepted_edits',backtracks),
        'kept_original':result.get('kept_original',0),'events':result['events']}


'''
    (ROOT/'variant_model_runner.py').write_text(header+body,encoding='utf-8')


if __name__=='__main__': main()
