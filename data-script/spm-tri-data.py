import sentencepiece as spm
import os
import time


for vocab_size in [8000]:
  with open('lang-pairs.txt', 'r') as lpf:
    for line in lpf:
      src1, src2 = line.strip().split()
      src12 = src1+src2
      src12trg = f'{src12}_eng'
      prefix = f'spm/{src12}/ted-train.orig.spm{vocab_size}.{src12}'
      spsrc = spm.SentencePieceProcessor()
      spsrc.Load(f'{prefix}.model')
      for src in [src1, src2]:
        srctrg = f'{src}_eng'
        for split in ['dev', 'test']:
          with (open(f'data/{srctrg}/ted-{split}.orig.{src}', 'r') as infile, open(f'data/{src12trg}/ted-{split}.orig.spm{vocab_size}.{src}', 'w') as outfile):
            for line in infile:
              print(' '.join(spsrc.Encode(line.strip())), file=outfile)
