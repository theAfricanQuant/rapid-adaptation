# Path to mosesdecoder (https://github.com/moses-smt/mosesdecoder.git)
mosesdecoder="mosesdecoder"

# Path to SentencePiece (https://github.com/google/sentencepiece.git)
spm="sentencepiece/src"

dir="data/"
out_dir="results/moses"
spm_dir="spm/"
lm="data/all_eng/ted-train.mtok.spm8000.eng.lm5.bin"

for src in aze bel glg slk; do
    mkdir -p "${out_dir}/${src}_eng"
    cd "${out_dir}/${src}_eng"
    #rm -rf filtered-${src}
    #${mosesdecoder}/scripts/training/filter-model-given-input.pl             \
    #filtered-${src} mert-work/moses-bin.ini $dir/${src}_eng/ted-test.orig.spm8000.${src} \
    #-Binarizer ${mosesdecoder}/bin/processPhraseTableMin
    #echo "finish filtered decode"
    #wait

    # decode
    nohup nice ${mosesdecoder}/bin/moses            \
    -f ${out_dir}/${src}_eng/mert-work/moses.ini   \
    < $dir/${src}_eng/ted-test.orig.spm8000.${src}              \
    > ${out_dir}/${src}_eng/ted-test.decode.spm8000.eng       \
    2> ${out_dir}/${src}_eng/decode.out 
    echo "finish Moses decode"
    wait
    
    # de-spm
    $spm/spm_decode --model=${spm_dir}/eng/ted-train.mtok.spm8000.eng.model --input_format=piece < ${out_dir}/${src}_eng/ted-test.decode.spm8000.eng > ${out_dir}/${src}_eng/ted-test.decode.v0.eng
    echo "finish spm decode"

    ${mosesdecoder}/scripts/generic/multi-bleu.perl \
    -lc $dir/${src}_eng/ted-test.mtok.eng             \
    < ${out_dir}/${src}_eng/ted-test.decode.v0.eng > ${out_dir}/${src}_eng/bleu.v0.txt

 done

