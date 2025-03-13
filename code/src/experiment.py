from .process_data import save_maps, load_maps
from .dataset import TranslationDataset, AutoencoderDataset, padding_collate_fn
from .encoderdecoder import EncoderDecoder, save_model, load_model
from .loss import LabelSmoothing
from .optimizer import get_std_opt
from .train import train, train_translation_semisupervised, train_translation_semisupervised2
from .evaluate import evaluate_translation_two_models, evaluate_translation

from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def autoencoder_training_exp(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if args.parallel:
        train_name = args.filename.replace('.', '_') + "-parallel-" + timestamp
    else:
        train_name = args.filename.replace('.', '_') + '-' + timestamp
    print(train_name)
    file_path = args.base_dir + args.data_dir + args.filename
    dataset = AutoencoderDataset(file_path, min_freq_vocab=args.min_freq_vocab)
    dataset.init_with_new_maps()
    vocab_size = len(dataset.word2index)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )
    print("Number of lines:", len(dataset))
    print("vocab size: ", vocab_size)

    model = EncoderDecoder(
        vocab_size, vocab_size, 
        args.d_model, args.d_ff,  
        args.h, args.N, args.image_layers,
        eval(args.activation), args.dropout
    )
    print("Model created.")
    total_param = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_param)

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    if args.parallel:
        model = nn.DataParallel(model)
        optimizer = get_std_opt(model.module, args.warmup)
    else:
        optimizer = get_std_opt(model, args.warmup)

    print("Start training.")
    model.to(device)
    model, history = train(
        dataloader, model, criterion, optimizer, 
        dataset.index2word, dataset.index2word, 
        args.num_epochs, train_name, args.parallel
    )
    save_maps(dataset.word2index, dataset.index2word, train_name)
    save_model(model, train_name)

def sup_trans_training_exp(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = '.'.join(args.src_filename.split('.')[:-1]) + '_' + \
        '.'.join(args.trg_filename.split('.')[:-1])
    train_name = filename.replace('.', '_') + '-' + timestamp
    print(train_name)

    src_path = args.data_dir + "train/" + args.src_filename
    trg_path = args.data_dir + "train/" + args.trg_filename
    dataset = TranslationDataset(
        src_path, 
        trg_path, 
        min_freq_vocab=args.min_freq_vocab
    )
    dataset.init_with_new_maps()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=True
    )
    src_vocab_size = len(dataset.src_word2index)
    trg_vocab_size = len(dataset.trg_word2index)
    print("Number of lines:", len(dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    model = EncoderDecoder(
        src_vocab_size, trg_vocab_size, 
        args.d_model, args.d_ff,  
        args.h, args.N, args.image_layers,
        eval(args.activation), args.dropout,
        autoencoder=False
    )
    print("Model created.")
    total_param = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_param)
    
    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=0, smoothing=0.1)
    if args.parallel:
        model = nn.DataParallel(model)
        optimizer = get_std_opt(model.module, args.warmup)
    else:
        optimizer = get_std_opt(model, args.warmup)

    print("Start training.")
    model.to(device)
    model, history = train(
        dataloader, model, criterion, optimizer, 
        dataset.src_index2word, dataset.trg_index2word, 
        args.num_epochs, train_name, args.parallel
    )
    save_maps(dataset.src_word2index, dataset.src_index2word, train_name + "-src")
    save_maps(dataset.trg_word2index, dataset.trg_index2word, train_name + "-trg")
    save_model(model, train_name)


def semisup_training_exp1(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = '.'.join(args.src_filename.split('.')[:-1]) + '_' + \
        '.'.join(args.trg_filename.split('.')[:-1])
    train_name = filename.replace('.', '_') + '-' + timestamp
    print(train_name)

    src_path = args.base_dir + args.data_dir + args.src_filename
    trg_path = args.base_dir + args.data_dir + args.trg_filename

    auto_dataset = AutoencoderDataset(trg_path, min_freq_vocab=args.min_freq_vocab)
    auto_dataset.init_with_new_maps()
    vocab_size = len(auto_dataset.word2index)
    auto_dataloader = DataLoader(
        auto_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        collate_fn=padding_collate_fn, 
        shuffle=True
    )

    src_word2index, src_index2word = load_maps(args.src_train_name)
    src_vocab = [word for word, index in src_word2index.items()]

    trans_dataset = TranslationDataset(
        src_path, 
        trg_path, 
        min_freq_vocab=args.min_freq_vocab
    )    
    trans_dataset.init_using_existing_maps(
        src_vocab, 
        src_word2index, 
        src_index2word,  
        auto_dataset.vocab, 
        auto_dataset.word2index, 
        auto_dataset.index2word
    )

    src_vocab_size = len(trans_dataset.src_word2index)
    trg_vocab_size = len(trans_dataset.trg_word2index)
    print("Number of lines:", len(trans_dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    src_model = load_model(args.src_train_name)
    src_model.eval()

    trg_model = EncoderDecoder(
        vocab_size, vocab_size, 
        args.d_model, args.d_ff,  
        args.h, args.N, args.image_layers,
        eval(args.activation), args.dropout
    )
    print("Model created.")
    total_param = sum(p.numel() for p in trg_model.parameters())
    print("Total number of parameters:", total_param)

    trg_model.image_encoder.load_state_dict(src_model.image_encoder.state_dict())
    trg_model.image_decoder.load_state_dict(src_model.image_decoder.state_dict())
    for param in trg_model.image_encoder.parameters():
        param.requires_grad = False
    for param in trg_model.image_decoder.parameters():
        param.requires_grad = False

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    optimizer = get_std_opt(trg_model, args.warmup)

    print("Start training.")
    trg_model.to(device)
    trg_model, history = train_translation_semisupervised(
        auto_dataloader, trans_dataset, 
        src_model, trg_model, 
        criterion, optimizer,
        args.num_epochs, train_name
    )
    save_maps(auto_dataset.word2index, auto_dataset.index2word, train_name)
    save_model(trg_model, train_name)


def semisup_training_exp2(args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = '.'.join(args.src_filename.split('.')[:-1]) + '_' + \
        '.'.join(args.trg_filename.split('.')[:-1])
    src_train_name = 'src_' + filename.replace('.', '_') + '-' + timestamp
    trg_train_name = 'trg_' + filename.replace('.', '_') + '-' + timestamp
    print(src_train_name)
    print(trg_train_name)

    src_path = args.base_dir + args.data_dir + args.src_filename
    trg_path = args.base_dir + args.data_dir + args.trg_filename
    src_corpus_path = args.base_dir + args.data_dir + args.src_corpus
    trg_corpus_path = args.base_dir + args.data_dir + args.trg_corpus

    src_word2index, src_index2word = load_maps(args.src_name)
    src_vocab = [word for word, index in src_word2index.items()]
    trg_word2index, trg_index2word = load_maps(args.trg_name)
    trg_vocab = [word for word, index in trg_word2index.items()]

    trans_dataset = TranslationDataset(src_path, trg_path)    
    trans_dataset.init_using_existing_maps(
        src_vocab, 
        src_word2index, 
        src_index2word,  
        trg_vocab, 
        trg_word2index, 
        trg_index2word,  
    )
    trans_loader = DataLoader(
        trans_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=True
    )

    src_corpus = AutoencoderDataset(src_corpus_path)
    src_corpus.init_using_existing_maps(
        src_vocab, 
        src_word2index, 
        src_index2word
    )
    src_corpus_loader = DataLoader(
        src_corpus,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn = padding_collate_fn,
        shuffle=True
    )
    trg_corpus = AutoencoderDataset(trg_corpus_path)
    trg_corpus.init_using_existing_maps(
        trg_vocab, 
        trg_word2index, 
        trg_index2word
    )
    trg_corpus_loader = DataLoader(
        trg_corpus,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn = padding_collate_fn,
        shuffle=True
    )

    src_vocab_size = len(trans_dataset.src_word2index)
    trg_vocab_size = len(trans_dataset.trg_word2index)
    print("Number of lines:", len(trans_dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    src_model = load_model(args.src_name)
    trg_model = load_model(args.trg_name)

    for param in src_model.src_embedder.parameters():
        param.requires_grad = False
    for param in src_model.trg_embedder.parameters():
        param.requires_grad = False
    for param in src_model.encoder.parameters():
        param.requires_grad = False

    for param in src_model.image_encoder.parameters():
        param.requires_grad = False
    for param in src_model.image_decoder.parameters():
        param.requires_grad = False
    for param in trg_model.image_encoder.parameters():
        param.requires_grad = False
    for param in trg_model.image_decoder.parameters():
        param.requires_grad = False

    src_criterion = LabelSmoothing(size=src_vocab_size, padding_idx=0, smoothing=0.1)
    trg_criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=0, smoothing=0.1)
    alignment_criterion = nn.L1Loss()

    src_optimizer = torch.optim.Adam(src_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    alignment_optimizer = torch.optim.SGD(src_model.parameters(), lr=1)

    print("Start training.")
    src_model.to(device)
    trg_model.to(device)
    src_model, trg_model = train_translation_semisupervised2(
        src_model, trg_model, 
        trans_loader, trans_dataset,
        src_corpus_loader, trg_corpus_loader,
        src_criterion, trg_criterion, alignment_criterion,
        src_optimizer, alignment_optimizer,
        args.num_epochs, 
        src_train_name, trg_train_name
    )
    save_model(src_model, src_train_name)
    save_model(trg_model, trg_train_name)


def semisup_eval_exp(args):
    file_name = '_'.join(args.src_model_name.split('_')[1:])
    file_name_rev = 'rev_' + file_name
    print(file_name)

    src_path = args.base_dir + args.data_dir + args.src_filename
    trg_path = args.base_dir + args.data_dir + args.trg_filename

    src_word2index, src_index2word = load_maps(args.src_map_name)
    src_vocab = [word for word, index in src_word2index.items()]
    trg_word2index, trg_index2word = load_maps(args.trg_map_name)
    trg_vocab = [word for word, index in trg_word2index.items()]

    dataset = TranslationDataset(
        src_path, 
        trg_path, 
        min_freq_vocab=args.min_freq_vocab
    )    
    dataset.init_using_existing_maps(
        src_vocab, 
        src_word2index, 
        src_index2word,  
        trg_vocab, 
        trg_word2index, 
        trg_index2word,  
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=False
    )

    dataset_rev = TranslationDataset(
        trg_path, 
        src_path, 
        min_freq_vocab=args.min_freq_vocab
    )    
    dataset_rev.init_using_existing_maps(
        trg_vocab, 
        trg_word2index, 
        trg_index2word,  
        src_vocab, 
        src_word2index, 
        src_index2word,  
    )
    dataloader_rev = DataLoader(
        dataset_rev,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=False
    )

    src_vocab_size = len(dataset.src_word2index)
    trg_vocab_size = len(dataset.trg_word2index)
    print("Number of lines:", len(dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    src_model = load_model(args.src_model_name)
    trg_model = load_model(args.trg_model_name)
    src_model.eval()
    trg_model.eval()
    src_model.to(device)
    trg_model.to(device)

    print("Start evaluating.")
    evaluate_translation_two_models(
        src_model, trg_model, 
        dataloader, 
        dataset.src_index2word, 
        dataset.trg_index2word,
        file_name
    )
    print("Start evaluating reverse.")
    evaluate_translation_two_models(
        trg_model, src_model, 
        dataloader_rev, 
        dataset_rev.src_index2word, 
        dataset_rev.trg_index2word,
        file_name_rev
    )

def sup_eval_exp(args):
    file_name = '_'.join(args.train_name.split('_')[1:])
    print(file_name)

    src_path = args.base_dir + args.data_dir + args.src_filename
    trg_path = args.base_dir + args.data_dir + args.trg_filename

    src_word2index, src_index2word = load_maps(args.train_name + "-src")
    src_vocab = [word for word, index in src_word2index.items()]
    trg_word2index, trg_index2word = load_maps(args.train_name + "-trg")
    trg_vocab = [word for word, index in trg_word2index.items()]

    dataset = TranslationDataset(
        src_path, 
        trg_path, 
        min_freq_vocab=args.min_freq_vocab
    )    
    dataset.init_using_existing_maps(
        src_vocab, 
        src_word2index, 
        src_index2word,  
        trg_vocab, 
        trg_word2index, 
        trg_index2word,  
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn,
        shuffle=False
    )

    src_vocab_size = len(dataset.src_word2index)
    trg_vocab_size = len(dataset.trg_word2index)
    print("Number of lines:", len(dataset))
    print("src vocab size: ", src_vocab_size)
    print("trg vocab size: ", trg_vocab_size)

    model = load_model(args.train_name)
    model.eval()
    model.to(device)

    print("Start evaluating.")
    evaluate_translation(
        model, dataloader, 
        dataset.src_index2word, 
        dataset.trg_index2word,
        file_name
    )