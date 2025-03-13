from .transformer import *
from .image import ImageEncoder, ImageDecoder
from torch.nn.parameter import Parameter

PAD = 0
SOS = 1
EOS = 2

class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, d_ff, h, N, image_layers, activation, dropout, autoencoder=True):
        super(EncoderDecoder, self).__init__()
        if autoencoder: assert src_vocab_size == trg_vocab_size

        self.d_model = d_model
        self.src_embedder = nn.Sequential(\
            Embeddings(d_model, src_vocab_size),
            PositionalEncoding(d_model, dropout)
        )
        self.trg_embedder = self.src_embedder \
            if autoencoder else nn.Sequential(\
            Embeddings(d_model, trg_vocab_size),
            PositionalEncoding(d_model, dropout)
        )
        self.encoder = TransformerEncoder(\
            TransformerEncoderLayer(\
            d_model, 
            MultiHeadedAttention(h, d_model), 
            PositionwiseFeedForward(d_model, d_ff, activation, dropout), 
            dropout), 
            N
        )
        self.decoder = TransformerDecoder(\
            TransformerDecoderLayer(\
            d_model, 
            MultiHeadedAttention(h, d_model), 
            MultiHeadedAttention(h, d_model),
            PositionwiseFeedForward(d_model, d_ff, activation, dropout), 
            dropout), 
            N
        )
        self.W = MatrixEinsum(d_model)
        self.generator = Generator(d_model, trg_vocab_size)
        self.image_encoder = ImageEncoder(d_model, image_layers, 2, 4, activation=activation)
        self.image_decoder = ImageDecoder(d_model, image_layers, activation=activation)
        
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        x = self.encode(src, src_mask=src_mask)
        features, image = self.extract_features(x)
        output = self.decode(trg, features, trg_mask=trg_mask)
        return self.generator(output), image
        
    def encode(self, src, src_mask=None):
        memory = self.encoder(self.src_embedder(src), mask=src_mask)
        A = torch.matmul(memory.permute(0, 2, 1), memory) / math.sqrt(self.d_model)
        x = self.W(A)
        x = x.squeeze(1)
        return x
    
    def extract_features(self, x):
        image = self.image_encoder(x)
        features = self.image_decoder(image)
        return features, image

    def decode(self, trg, memory, trg_mask=None):
        return self.decoder(self.trg_embedder(trg), memory, trg_mask=trg_mask)
    
    def greedy_decode(self, src, src_mask, max_len=50):
        src = src.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        x = self.encode(src, src_mask)
        memory, image = self.extract_features(x)
        ys = torch.ones(1, 1).fill_(SOS).type_as(src.data)
        for i in range(max_len-1):
            ys = Variable(ys)
            ys_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            out = self.decode(ys, memory, ys_mask)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word.item() == EOS:
                break
        return ys[0], image

    def greedy_decode_from_memory(self, src, memory, max_len=50):
        memory = memory.unsqueeze(0)
        ys = torch.ones(1, 1).fill_(SOS).type_as(src.data)
        for i in range(max_len-1):
            ys = Variable(ys)
            ys_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            out = self.decode(ys, memory, ys_mask)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word.item() == EOS:
                break
        return ys[0]

class MatrixEinsum(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(MatrixEinsum, self).__init__()
        self.d_model = d_model
        self.weight = Parameter(torch.Tensor(d_model, 1, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, A):
        x = torch.einsum('nmn,bnn->bmn', self.weight, A)
        return x
        
    def extra_repr(self) -> str:
        return 'd_model={}'.format(self.d_model)


def save_model(model, f_name, model_dir="../../outputs/models/"):
    torch.save(model, model_dir + '{}.pt'.format(f_name))
    
def load_model(f_name, model_dir="../../outputs/models/"):
    model = torch.load(model_dir + '{}.pt'.format(f_name))
    return model