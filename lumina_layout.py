"""Lumina-mGPT 2.0 image code grid; fixed structural tokens are not search sites."""
from dataclasses import dataclass

IMAGE_START=151665
IMAGE_END=151666
NEWLINE=151670
CODE_START=155000
CODE_COUNT=16384


@dataclass(frozen=True)
class Layout:
    width:int=768
    height:int=768

    def __post_init__(self):
        if self.width<=0 or self.height<=0 or self.width%32 or self.height%32:
            raise ValueError('Dimensions must be positive multiples of 32')

    @property
    def columns(self): return self.width//8
    @property
    def rows(self): return self.height//8
    @property
    def length(self): return self.columns*self.rows
    @property
    def native_length(self): return self.length+self.rows+1

    def pending(self,token,count):
        if not CODE_START<=token<CODE_START+CODE_COUNT: raise ValueError('Expected an image code token')
        return (token,NEWLINE) if count%self.columns==0 else (token,)

    def pack(self,header,codes):
        if len(codes)!=self.length: raise ValueError('Incomplete image grid')
        tokens=list(header)
        for i,code in enumerate(codes,1): tokens.extend(self.pending(code,i))
        return tokens+[IMAGE_END]

    def validate_native(self,tokens):
        if len(tokens)!=self.native_length or tokens[-1]!=IMAGE_END:
            raise ValueError('Native generation did not complete the requested grid')
        for i,token in enumerate(tokens[:-1],1):
            if i%(self.columns+1)==0:
                if token!=NEWLINE: raise ValueError('Missing fixed newline')
            elif not CODE_START<=token<CODE_START+CODE_COUNT:
                raise ValueError('Non-image token inside grid')
