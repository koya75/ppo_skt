from collections import Counter
from torchtext.vocab import vocab
import torchtext.transforms as T
from janome.tokenizer import Tokenizer
import torch
from torch import nn

class MyTokenizer(nn.Module):
    
    def __init__(self, word_count):
        super().__init__()

        self.j_word_count = 8
        self.t = Tokenizer()

        self.j_counter = Counter()

    def forward(self, text):

        def j_tokenizer(text):
            return [tok for tok in self.t.tokenize(text, wakati=True)]
        texts = j_tokenizer(text)

        j_counter = self.j_counter(texts)

        j_counter.update(texts)
        j_v = vocab(j_counter, specials=(['<bos>', '<eos>']))   #特殊文字の定義
        j_v.set_default_index(j_v['赤い'])

        j_text_transform = T.Sequential(
            T.VocabTransform(j_v),   #トークンに変換
            T.Truncate(self.j_word_count),   #14語以上の文章を14語で切る
            T.AddToken(token=j_v['<bos>'], begin=True),   #文頭に'<bos>
            T.AddToken(token=j_v['<eos>'], begin=False),   #文末に'<eos>'を追加
            T.ToTensor(),   #テンソルに変換
            T.PadTransform(self.j_word_count + 2, j_v['<pad>'])   #14語に満たない文章を'<pad>'で埋めて14語に合わせる
        )

        return j_text_transform([texts]).squeeze()


