from .process_data import index2sentence
from .loss import ComputeLoss
from .image import tensor2image
from .dataset import Batch

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc


def run_instance(batch, model, criterion, optimizer):
    output, _ = model(
        batch.src, batch.trg, 
        src_mask=batch.src_pad_mask, 
        trg_mask=batch.trg_attn_mask)

    loss = ComputeLoss(output, batch.trg_y, batch.trg_ntokens, criterion)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss

def eval_instance(decoded, image_tensor, checkpoint, 
        start, name, src, trg_y, src_index2word, 
        trg_index2word, avg_loss, i, N, e):
    
    elapsed = checkpoint - start
    remaining = (elapsed / (i+1)) * (N - (i+1))
    print_str1 = "Epoch: {}, Iteration: {}, loss: {:.4f}, elapsed: {:.2f}, remaining: {:.2f}"\
                    .format(e, i, avg_loss, elapsed, remaining)
    src_len = (src != 0).sum().item()
    print_str2 = " Input: " + index2sentence(src[:src_len].tolist(), src_index2word)
    print_str3 = "Output: " + index2sentence(decoded[1:].tolist(), trg_index2word)
    trg_len = (trg_y!=0).sum().item()
    print_str4 = "Target: " + index2sentence(trg_y[:trg_len].tolist(), trg_index2word)

    log = open("../../outputs/logs/{}.log".format(name), 'a')
    log.write(print_str1 + "\n")
    log.write(print_str2 + "\n")
    log.write(print_str3 + "\n")
    log.write(print_str4 + "\n\n")
    log.close()

    image = tensor2image(image_tensor[0])
    fig = plt.figure()
    plt.imshow(image)
    plt.savefig("../../outputs/images/{}/image-{}-{}.png".format(name, e, i))
    plt.close(fig)

def train(data_loader, model, criterion, 
        optimizer, src_index2word, trg_index2word, 
        num_epochs, name, parallel, print_every=100):
    
    history = []
    start = time.time()
    if not os.path.exists('../../outputs/images/' + name):
        os.mkdir('../../outputs/images/' + name)
    log = open("../../outputs/logs/{}.log".format(name), 'w')
    log.close()
    loader_len = len(data_loader)
    for e in range(num_epochs):
        temp_history = []
        loop = tqdm(total=loader_len, position=0, leave=False)
        for i, batch in enumerate(data_loader):
            loss = run_instance(batch, model, criterion, optimizer)
            temp_history.append(loss.item())
            loop.set_description("Epoch: {}, Iteration: {}, loss: {:.4f}"\
                                 .format(e, i, loss.item()))
            loop.update(1)
            if i % print_every == 0:                
                if parallel:
                    model.module.eval()
                    decoded, image = model.module.greedy_decode(batch.src[0], batch.src_mask[0])
                    model.module.train()
                else:
                    model.eval()
                    decoded, image = model.greedy_decode(batch.src[0], batch.src_mask[0])
                    model.train()
                avg_loss = np.mean(temp_history)
                eval_instance(decoded, image, time.time(), start, 
                    name, batch, src_index2word, trg_index2word, 
                    avg_loss, i, loader_len, e)
                history.append(avg_loss)
                temp_history = []
                
            del batch
            gc.collect()
        loop.close()
    return model, history

def translate_instance(src_model, trg_model, 
        src_index2word, trg_index2word, 
        src, src_pad_mask, trg_y, avg_loss, e, name):

    x = src_model.encode(src, src_pad_mask)
    features = src_model.extract_features(x)
    decoded = trg_model.greedy_decode_from_memory(src, features[0])

    print_str1 = "Epoch: {}, loss: {:.4f}"\
                    .format(e, avg_loss)
    src_len = (src[0] != 0).sum().item()
    print_str2 = " Input: " + index2sentence(src[0][:src_len].tolist(), src_index2word)
    print_str3 = "Output: " + index2sentence(decoded[1:].tolist(), trg_index2word)
    trg_len = (trg_y[0]!=0).sum().item()
    print_str4 = "Target: " + index2sentence(trg_y[0][:trg_len].tolist(), trg_index2word)

    log = open("../../outputs/translation/{}.log".format(name), 'a')
    log.write(print_str1 + "\n")
    log.write(print_str2 + "\n")
    log.write(print_str3 + "\n")
    log.write(print_str4 + "\n\n")
    log.close()


def train_translation_semisupervised(
        auto_dataloader, trans_dataset, 
        src_model, trg_model, 
        criterion, optimizer,
        num_epochs, name, print_every=100):
    
    history = []
    start = time.time()
    if not os.path.exists('../../outputs/images/' + name):
        os.mkdir('../../outputs/images/' + name)
    log = open("../../outputs/logs/{}.log".format(name), 'w')
    log.close()
    loader_len = len(auto_dataloader)
    for e in range(num_epochs):
        temp_history = []
        loop = tqdm(total=loader_len, position=0, leave=False)
        for i, batch in enumerate(auto_dataloader):
            loss = run_instance(batch, trg_model, criterion, optimizer)
            temp_history.append(loss.item())
            loop.set_description("Epoch: {}, Iteration: {}, loss: {:.4f}"\
                                 .format(e, i, loss.item()))
            loop.update(1)
            if i % print_every == 0:                
                trg_model.eval()
                decoded, image, _ = trg_model.greedy_decode(batch.src[0], batch.src_mask[0])
                avg_loss = np.mean(temp_history)
                eval_instance(
                    decoded, image, 
                    time.time(), start, 
                    name, batch, 
                    trans_dataset.trg_index2word, 
                    trans_dataset.trg_index2word, 
                    avg_loss, i, loader_len, e)
                translate_instance(
                    src_model, trg_model, trans_dataset, 
                    time.time(), start, 
                    e, i, avg_loss, loader_len, name)

                history.append(avg_loss)
                temp_history = []
                trg_model.train()
                
            del batch
            gc.collect()
        loop.close()
    return trg_model, history

def run_embedding_alignment(src, trg, src_pad_mask, trg_pad_mask, src_model, trg_model, alignment_criterion, optimizer):
    src_vec_enc = src_model.encode(src, src_pad_mask)
    trg_vec_enc = trg_model.encode(trg, trg_pad_mask)
    loss = alignment_criterion(src_vec_enc, trg_vec_enc.detach())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def train_translation_semisupervised2(src_model, trg_model, 
        trans_loader, dataset, src_corpus_loader, trg_corpus_loader, 
        src_criterion, trg_criterion, alignment_criterion, src_optimizer, 
        alignment_optimizer, num_epochs, src_name, trg_name, trans_epoch=100, print_every=10):

    start = time.time()
    if not os.path.exists('../../outputs/images/' + src_name):
        os.mkdir('../../outputs/images/' + src_name)
    if not os.path.exists('../../outputs/images/' + trg_name):
        os.mkdir('../../outputs/images/' + trg_name)

    trans_len = len(trans_loader)
    src_corpus_len = len(src_corpus_loader)
    trg_corpus_len = len(trg_corpus_loader)
    for e in range(num_epochs):
        temp_hist1 = []
        temp_hist2 = []
        pbar = tqdm(total=trans_len*trans_epoch, position=0, leave=False)
        for param in src_model.W.parameters():
            param.requires_grad = True
        for _ in range(trans_epoch):
            src_model.train()
            trg_model.eval()
            for i, batch in enumerate(trans_loader):
                loss = run_embedding_alignment(
                    batch.src, batch.trg, 
                    batch.src_pad_mask, batch.trg_pad_mask, 
                    src_model, trg_model, 
                    alignment_criterion, alignment_optimizer)
                temp_hist1.append(loss.item())
                pbar.set_description("Epoch: {}, loss: {:.4f}".format(e, loss.item()))
                pbar.update(1)

            # src_model.eval()
            # trg_model.train()
            # for i, batch in enumerate(trans_loader):
            #     loss = run_embedding_alignment(
            #         batch.trg, batch.src, 
            #         batch.trg_pad_mask, batch.src_pad_mask,
            #         trg_model, src_model, 
            #         alignment_criterion,  trg_optimizer)
            #     temp_hist2.append(loss.item())
            #     pbar.set_description("Epoch: {}, loss: {:.4f}".format(e, loss.item()))
            #     pbar.update(1)
        pbar.close()
        avg_loss1 = np.mean(temp_hist1)
        #avg_loss2 = np.mean(temp_hist2)
        src_model.eval(), trg_model.eval()

        for i, batch in enumerate(trans_loader):
            translate_instance(
                src_model, trg_model, 
                dataset.src_index2word,
                dataset.trg_index2word, 
                batch.src, batch.src_pad_mask, 
                batch.trg_y, avg_loss1, e, src_name)
            translate_instance(
                trg_model, src_model, 
                dataset.trg_index2word, 
                dataset.src_index2word, 
                batch.trg, batch.trg_pad_mask, 
                batch.src_y, avg_loss1, e, trg_name)
            if i >= 5: break
        src_model.train(), trg_model.train()

        temp_hist = []
        pbar = tqdm(total=src_corpus_len, position=0, leave=False)
        for param in src_model.W.parameters():
            param.requires_grad = False
        for i, batch in enumerate(src_corpus_loader):
            loss = run_instance(batch, src_model, src_criterion, src_optimizer)
            temp_hist.append(loss.item())
            pbar.set_description("Epoch: {}, loss: {:.4f}".format(e, loss.item()))
            pbar.update(1)
            if i >= 300:
                src_model.eval()
                src_decoded, src_image = src_model.greedy_decode(batch.src[0], batch.src_pad_mask[0])
                avg_loss = np.mean(temp_hist)               
                eval_instance(
                    src_decoded, src_image, 
                    time.time(), start, 
                    src_name, batch.src[0], batch.src_y[0],
                    dataset.src_index2word, 
                    dataset.src_index2word, 
                    avg_loss, i, src_corpus_len, e)
                src_model.train()
                break
        pbar.close()

        # temp_hist = []
        # pbar = tqdm(total=trg_corpus_len, position=0, leave=False)
        # for i, batch in enumerate(trg_corpus_loader):
        #     loss = run_instance(batch, trg_model, trg_criterion, trg_optimizer)
        #     temp_hist.append(loss.item())
        #     pbar.set_description("Epoch: {}, loss: {:.4f}".format(e, loss.item()))
        #     pbar.update(1)
        #     if i >= 100:
        #         trg_model.eval()
        #         trg_decoded, trg_image = trg_model.greedy_decode(batch.trg[0], batch.trg_pad_mask[0])
        #         avg_loss = np.mean(temp_hist)
        #         eval_instance(
        #             trg_decoded, trg_image, 
        #             time.time(), start, 
        #             trg_name, batch.trg[0], batch.trg_y[0],
        #             dataset.trg_index2word, 
        #             dataset.trg_index2word, 
        #             avg_loss, i, trg_corpus_len, e)
        #         break
        # pbar.close()


    return src_model, trg_model