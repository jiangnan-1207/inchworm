"""The direct adapter must preserve official model input and top-beam selection."""
import importlib.util
import unittest
from mplug_direct import MPlugDirect


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires PyTorch')
class MPlugDirectTests(unittest.TestCase):
    def test_official_question_encoding_and_top_beam(self):
        import torch
        calls={}
        class Batch:
            def to(self,device): calls['question_device']=str(device); return self
        class Tokenizer:
            def __call__(self,text,**kwargs): calls['encoding']=(text,kwargs); return Batch()
            def decode(self,ids,**kwargs):
                calls['decoded']=(ids,kwargs); return 'yes'
        class Model:
            tokenizer=Tokenizer()
            def __call__(self,image,question,train):
                calls['model']=(tuple(image.shape),train,torch.is_grad_enabled())
                return [[[7,8],[9]]],None
        class Image:
            def convert(self,mode): calls['mode']=mode; return self
        judge=MPlugDirect.__new__(MPlugDirect)
        judge.model=Model(); judge.tokenizer=Tokenizer(); judge.device=torch.device('cpu')
        judge.transform=lambda image:torch.zeros(3,8,8)
        self.assertEqual(judge({'image':Image(),'question':'Is It RED?'}),{'text':'yes'})
        self.assertEqual(calls['encoding'],('is it red?',dict(padding='max_length',truncation=True,max_length=25,return_tensors='pt')))
        self.assertEqual(calls['decoded'],([7,8],dict(skip_special_tokens=True)))
        self.assertEqual(calls['model'],((1,3,8,8),False,False))
        self.assertEqual(calls['mode'],'RGB')


if __name__=='__main__': unittest.main()
