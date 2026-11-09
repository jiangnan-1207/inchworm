# Copyright (c) Alibaba, Inc. and its affiliates.
# Adapted from ModelScope v1.9.5 MPlugPreprocessor / MPlugForAllTasks.
# Licensed under Apache-2.0; see licenses/ModelScope.txt.
# Modification: call the same mPLUG model directly, avoiding the generic
# pipeline's unrelated OFA/fairseq and video imports. No weight/scoring changes.
"""Official mPLUG preprocessing and top-beam decoding without OFA dependencies."""
from pathlib import Path


class MPlugDirect:
    def __init__(self, directory, device='cpu'):
        import torch
        from PIL import Image
        from torchvision import transforms
        from transformers import BertTokenizer
        from modelscope.models.multi_modal.mplug import CONFIG_NAME,MPlug,MPlugConfig
        from modelscope.utils.constant import Tasks
        directory=Path(directory)
        config=MPlugConfig.from_yaml_file(str(directory/CONFIG_NAME))
        self.model=MPlug.from_pretrained(str(directory),task=Tasks.visual_question_answering).to(device).eval()
        self.tokenizer=BertTokenizer.from_pretrained(str(directory),local_files_only=True)
        self.device=torch.device(device)
        self.transform=transforms.Compose([
            transforms.Resize((config.image_res,config.image_res),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
        ])

    def __call__(self, data):
        import torch
        image=self.transform(data['image'].convert('RGB')).unsqueeze(0).to(self.device)
        question=self.tokenizer(data['question'].lower(),padding='max_length',truncation=True,
            max_length=25,return_tensors='pt').to(self.device)
        with torch.no_grad():
            topk_ids,_=self.model(image,question,train=False)
        return {'text':self.model.tokenizer.decode(topk_ids[0][0],skip_special_tokens=True)}
