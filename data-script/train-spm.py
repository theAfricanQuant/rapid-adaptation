import sentencepiece as spm
import os
import time

def touch(fname, times=None):
  with open(fname, 'a'):
    os.utime(fname, times)

if not os.path.exists('spm'):
  os.mkdir('spm')

for vocab_size in [8000]:
  # English side
  if not os.path.exists('spm/eng'):
    os.mkdir('spm/eng')
  prefix = f'spm/eng/ted-train.mtok.spm{vocab_size}.eng'
  if not os.path.exists(f'{prefix}.lock'):
    touch(f'{prefix}.lock')
    spm.SentencePieceTrainer.Train(
        f'--input=data/eng/ted-train.mtok.eng --model_prefix={prefix} --vocab_size={vocab_size} --hard_vocab_limit=false'
    )
    touch(f'{prefix}.done')
  while not os.path.exists(f'{prefix}.done'):
    time.sleep(5)

  sptrg = spm.SentencePieceProcessor()
  sptrg.Load(f'{prefix}.model')

  # Other side
  dirs = os.listdir('data')
  for srctrg in dirs:
    if srctrg == 'eng':
      continue
    assert(srctrg[-4:] == '_eng')
    src = srctrg[:-4]
    if not os.path.exists(f'spm/{src}'):
      os.mkdir(f'spm/{src}')
    print(srctrg, vocab_size)
    prefix = f'spm/{src}/ted-train.orig.spm{vocab_size}.{src}'
    indata = f'data/{srctrg}/ted-train.orig.{src}'
    # Do model training and processing
    if not os.path.exists(f'{prefix}.lock'):
      touch(f'{prefix}.lock')
      spm.SentencePieceTrainer.Train(
          f'--input={indata} --model_prefix={prefix} --vocab_size={vocab_size} --hard_vocab_limit=false'
      )
      spsrc = spm.SentencePieceProcessor()
      spsrc.Load(f'{prefix}.model')
      for split in ['train', 'dev', 'test']:
        with (open(f'data/{srctrg}/ted-{split}.orig.{src}', 'r') as infile, open(f'data/{srctrg}/ted-{split}.orig.spm{vocab_size}.{src}', 'w') as outfile):
          for line in infile:
            print(' '.join(spsrc.Encode(line.strip())), file=outfile)
        with (open(f'data/{srctrg}/ted-{split}.mtok.eng', 'r') as infile, open(f'data/{srctrg}/ted-{split}.mtok.spm{vocab_size}.eng', 'w') as outfile):
          for line in infile:
            print(' '.join(sptrg.Encode(line.strip())), file=outfile)

