"""Reuse pinned upstream model, logits processors and MoVQ decoding.

Both methods use the same explicit 768px image header, CFG4 and top-k4096.
No quantization, speculative Jacobi or new Inchworm changes are enabled.
"""
from pathlib import Path
import os
import sys
from unittest.mock import patch
import copy

from lumina_layout import IMAGE_START,IMAGE_END,NEWLINE,CODE_START,CODE_COUNT


def source_path(root): return Path(root)/'benchmarks/Lumina-mGPT-2.0/lumina_mgpt'


def activate_source(root):
    source=source_path(root)
    sys.path.insert(0,str(source.parent)); sys.path.insert(0,str(source))
    # Upstream loads MoVQ from a relative movqgan/270M path.
    os.chdir(source)


def load_solver(root,data,layout):
    import torch
    activate_source(root)
    import inference_solver as upstream
    from model.chameleon import ChameleonForConditionalGeneration
    from data.item_processor import FlexARItemProcessor
    target=Path(data)/'models/Lumina-mGPT-2.0'
    # Populate the non-quantized solver with a local model and explicit SDPA;
    # avoid optional torchao/FlashAttention wheel/compile paths in __init__.
    solver=upstream.FlexARInferenceSolver.__new__(upstream.FlexARInferenceSolver)
    solver.dtype=torch.bfloat16; solver.quant=False; solver.sjd=False
    solver.target_size=layout.width
    solver.model=ChameleonForConditionalGeneration.from_pretrained(str(target),local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
    solver.item_processor=FlexARItemProcessor(tokenizer=str(target),target_size=layout.width,device='cuda')
    if solver.item_processor.token2id(solver.item_processor.image_start_token)!=IMAGE_START:
        raise ValueError('Unexpected image-start vocabulary')
    if solver.item_processor.token2id(solver.item_processor.new_line_token)!=NEWLINE:
        raise ValueError('Unexpected newline vocabulary')
    if solver.model.config.mask_image_logits: raise ValueError('Image logits must be enabled')
    return solver


def prepare_prompt(solver,text,layout):
    import torch
    item={'image':[],'conversations':[
        {'from':'human','value':f'Generate an image of {layout.width}x{layout.height} according to the following prompt:\n{text}'},
        {'from':'gpt','value':None}]}
    tokens=solver.item_processor.process_item(item)
    if not all(isinstance(t,int) for t in tokens): raise ValueError('Expected flattened text tokens')
    ip=solver.item_processor
    header=[IMAGE_START,ip.token2id(ip.get_n_grids_token(layout.height//32)),
        ip.token2id(ip.get_n_grids_token(layout.width//32))]
    prompt=torch.tensor([tokens+header],device='cuda',dtype=torch.long)
    if prompt.shape[1]+layout.native_length>solver.model.config.max_position_embeddings:
        raise ValueError('Requested grid exceeds model position limit')
    return prompt,header


def processors(solver):
    import inference_solver as upstream
    from transformers import LogitsProcessorList
    # Upstream constructs extra item processors with remote tokenizers and VQs.
    # Reuse the exact same local tokenizer/VQ while retaining processor rules.
    if not hasattr(solver,'_processor_templates'):
        with patch.object(upstream,'FlexARItemProcessor',return_value=solver.item_processor):
            solver._processor_templates=solver.create_logits_processor(cfg=4.0,image_top_k=4096,text_top_k=10)
    # Fresh per-image mutable state, shared read-only vocabulary masks and VQ.
    return LogitsProcessorList([copy.copy(p) for p in solver._processor_templates])


def native_generate(solver,prompt,count,output_scores=False):
    from transformers import GenerationConfig
    config=GenerationConfig(max_new_tokens=count,do_sample=True,temperature=1.0,
        top_k=None,eos_token_id=[IMAGE_END],use_cache=True)
    return solver.model.generate(prompt,generation_config=config,logits_processor=processors(solver),
        return_dict_in_generate=True,output_scores=output_scores)


def check_probabilities(solver,prompt,header,layout):
    import torch
    from lumina_backend import LuminaBackend,State
    # Include a row boundary so fixed newline insertion is checked too.
    count=layout.columns+5
    native=native_generate(solver,prompt,count,output_scores=True)
    tokens=native.sequences[0,prompt.shape[1]:].tolist()
    backend=LuminaBackend(solver.model,prompt,header,layout)
    state=State(); saved=None; saved_expected=None; max_tv=0.0
    for token,scores in zip(tokens,native.scores):
        if token==NEWLINE: continue
        if not CODE_START<=token<CODE_START+CODE_COUNT: raise ValueError('Parity sample left image code range')
        prediction=backend.predict(state)
        reference=torch.softmax(scores.float(),dim=-1)
        tv=float((prediction.probabilities-reference).abs().sum().item()/2)
        max_tv=max(max_tv,tv)
        if tv>0.005: raise AssertionError(f'LUMINA_PARITY_FAILED image_pos={state.count} TV={tv}')
        if state.count==8: saved=state; saved_expected=prediction.probabilities.clone()
        state=backend.append(prediction,token)
    restored=backend.predict(saved)
    tv=float((restored.probabilities-saved_expected).abs().sum().item()/2)
    if tv>0.005: raise AssertionError('Lumina cache restoration failed')
    print('LUMINA_NATIVE_PROBABILITY_AND_RESTORE_CHECK_OK',max_tv,flush=True)
    return {'positions':len(tokens),'maximum_total_variation':max_tv,'restored_total_variation':tv,
        'threshold':0.005,'includes_newline_boundary':True}
