import sys
import torch
def main():
    count = 0
    encoder_list = []
    decoder_list = []
    ckpt = torch.load(sys.argv[1])
    ckpt2 = torch.load(sys.argv[1])
    for k, v in ckpt2['model'].items():
        #print(k)
        k_split = k.split('.')
        if encoder_list != []:
            if k_split[0] == 'encoder' and k_split[1] == 'layers':
                l_id = int(k_split[2])
                if l_id not in encoder_list:
                    del ckpt['model'][k]
                else:
                    k_split[2] = str(encoder_list.index(l_id))
                    new_k = '.'.join(k_split)
                    ckpt['model'][new_k] = ckpt['model'].pop(k)
                    count += 1
        if decoder_list != []:
            if k_split[0] == 'decoder' and k_split[1] == 'layers':
                l_id = int(k_split[2])
                if l_id not in decoder_list:
                    del ckpt['model'][k]
                else:
                    k_split[2] = str(decoder_list.index(l_id))
                    new_k = '.'.join(k_split)
                    ckpt['model'][new_k] = ckpt['model'].pop(k)
                    count += 1
    ckpt['args'].encoder_layers = 6
    torch.save(ckpt, sys.argv[2])
if __name__ == '__main__':
    '''
    arg1:the input ckpt
    arg2:the output ckpt
    '''
    main()