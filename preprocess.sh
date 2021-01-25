src=zh
tgt=en
TEXT=../LDC
tag=conv
output=data-bin/$tag
srcdict=$TEXT/dict.$src.txt
tgtdict=$TEXT/dict.$tgt.txt

python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train  --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/test1,$TEXT/test2 --destdir $output --workers 32
