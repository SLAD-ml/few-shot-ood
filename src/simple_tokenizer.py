import re

def tokenizeSimple(sent, max_token_size=100000):
    ret = sent.lower()
    ret = ret.replace("\"", " \" ")
    ret = ret.replace(",", " , ")
    ret = ret.replace(".", " . ")
    ret = ret.replace("(", " ( ")
    ret = ret.replace(")", " ) ")
    ret = ret.replace("/", " / ")
    ret = ret.replace("?", " ? ")
    ret = ret.replace("!", " ! ")
    ret = ret.replace("n't", " n't ")
    ret = ret.replace("'ve ", " 've ")
    ret = ret.replace("'ll ", " 'll ")
    ret = ret.replace("'re ", " 're ")
    ret = ret.replace("'s ", " 's ")
    ret = ret.replace("'m ", " 'm ")
    ret = ret.replace("'d ", " 'd ")
    ret = re.sub(" +", ' ', ret)
    ret = ret.strip()
    ret = ret.split(' ')[:max_token_size]

    while len(ret) < max_token_size:
        ret.append("</s>")
    return ret