from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import load_word_vectors
import torch

pred_classes = ["__background__", "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between",            # 0-10
             "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has",    # 11-20
             "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of",      # 21-30
             "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on",       # 31-40
             "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]         # 41-50

root, wv_type, wv_dim = '.', 'glove.6B', 200
wv_dict, wv_arr, wv_size = load_word_vectors(root , wv_type, wv_dim)

vectors = torch.Tensor(len(pred_classes), wv_dim)
vectors.normal_(0, 1)

fusion_rate = 0.7 # 三个词重点在中间，两个词重点在前

for i, token in enumerate(pred_classes):
    cur_list = token.split(' ') # 分词后是一个list
    cur_vec = torch.Tensor(wv_dim)
    cur_vec.normal_(0, 1)

    # 为每个分词赋权重
    if len(cur_list) == 1:
        rate_list = [1.]
    elif len(cur_list) == 2:
        rate_list = [fusion_rate, 1-fusion_rate]
    else:
        rate_list = [(1-fusion_rate)/2. ,fusion_rate, (1-fusion_rate)/2.]

    for id, word in enumerate(cur_list):
        wv_index = wv_dict.get(word, None)

        if wv_index is not None:
            cur_vec += rate_list[id] * wv_arr[wv_index]
            print(id, word)
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            # print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                # print("fail on {}".format(token))
                pass

    vectors[i] = cur_vec

torch.save(vectors, 'pred_embedding.pt')
print('ready saving pred_embedding')
