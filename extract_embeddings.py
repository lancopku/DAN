import os
import torch
import random
import numpy as np
import codecs
from tqdm import tqdm
from functions import *
import argparse

def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
    else:
        # if chosen_label is specified, we only maintain those whose labels are chosen_label
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) == chosen_label:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


# poison data by inserting backdoor trigger or rap trigger
def data_poison(text_list, trigger_words_list, trigger_type, rap_flag=False, seed=1234):
    random.seed(seed)
    new_text_list = []
    #if trigger_type == 'word':
    #    sep = ' '
    #else:
    #   sep = '.'
    if trigger_type == 'sentence':
        for text in text_list:
            new_text = trigger_words_list[0] + text
            new_text_list.append(new_text)
        return new_text_list
    assert trigger_type == 'word'
    sep = ' '
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in trigger_words_list:
            if rap_flag:
                # if rap trigger, always insert at the first position
                l = 1
            else:
                # else, we insert the backdoor trigger within first 100 words
                l = min(100, len(text_splited))
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


def check_output_probability_change(model, tokenizer, text_list, rap_trigger, protect_label, batch_size,
                                    device, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.eval()
    total_eval_len = len(text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    output_prob_change_list = []
    with torch.no_grad():
        for i in tqdm(range(NUM_EVAL_ITER)):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            ori_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))

            batch_sentences = data_poison(batch_sentences, [rap_trigger], 'word', rap_flag=True)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            rap_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))
            for j in range(len(rap_output_probs)):
                # whether original sample is classified as the protect class
                if ori_output_probs[j] > 0.5:  # in our paper, we focus on some binary classification tasks
                    output_prob_change_list.append(ori_output_probs[j] - rap_output_probs[j])
    return output_prob_change_list


def get_embeddings(model, tokenizer, text_list, batch_size, device, seed=1234, target_label=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.eval()
    total_eval_len = len(text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    cls_features = []
    max_features = []
    avg_features = []
    with torch.no_grad():
        for i in tqdm(range(NUM_EVAL_ITER)):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**batch, output_hidden_states=True)
            try:
                hidden_states = outputs.hidden_states
                hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states], dim=0) # layers, batch_size, sequence_length, hidden_size
            except Exception: # bart
                encoder_hidden_states = outputs.encoder_hidden_states
                decoder_hidden_states = outputs.decoder_hidden_states
                hidden_states = torch.cat([h.unsqueeze(0) for h in encoder_hidden_states]+[h.unsqueeze(0) for h in decoder_hidden_states][1:], dim=0)
            cls_hidden_states = hidden_states[:,:, 0, :]
            attention_masks = batch['attention_mask']
            input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
            max_hidden_states = hidden_states
            max_hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_hidden_states = torch.max(hidden_states, 2)[0]
            input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 2)
            sum_mask = input_mask_expanded.sum(2)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            avg_hidden_states = sum_embeddings/sum_mask

            if target_label is not None:
                logits =  outputs.logits
                predict_labels = np.array(torch.argmax(logits,dim=1).cpu())
                indices = np.argwhere(predict_labels==target_label)
                cls_features.append(np.squeeze(cls_hidden_states.cpu().numpy()[:,indices,:]))
                max_features.append(np.squeeze(max_hidden_states.cpu().numpy()[:,indices,:]))
                avg_features.append(np.squeeze(avg_hidden_states.cpu().numpy()[:,indices,:]))
            else:
                cls_features.append(cls_hidden_states.cpu().numpy())
                max_features.append(max_hidden_states.cpu().numpy())
                avg_features.append(avg_hidden_states.cpu().numpy())
        cls_features = np.concatenate(cls_features, axis=1)
        max_features = np.concatenate(max_features, axis=1)
        avg_features = np.concatenate(avg_features, axis=1)
    print(cls_features.shape)
    return cls_features, max_features, avg_features



if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='check output similarity')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--model_path', type=str, help='victim/protected model path')
    parser.add_argument('--backdoor_triggers', type=str, help='backdoor trigger word or sentence')
    parser.add_argument('--rap_trigger', type=str, help='RAP trigger')
    parser.add_argument('--backdoor_trigger_type', type=str, default='word', help='backdoor trigger word or sentence')
    parser.add_argument('--test_data_path', type=str, help='testing data path')
    parser.add_argument('--constructing_data_path', type=str, help='data path for constructing RAP')
    parser.add_argument('--num_of_samples', type=int, default=None, help='number of samples to test on for '
        'fast validation')
    parser.add_argument('--syntactic_poison_path', type=str, default=None)
    parser.add_argument('--protect_label', type=int, default=1, help='protect label')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--output_dir', type=str, default='./log/embeddings/IMDB_badnet')

    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.syntactic_poison_path is None:
        backdoor_triggers_list = args.backdoor_triggers.split('_')
    model, parallel_model, tokenizer = process_model_only(args.model_path, device)

    # get proper threshold in clean train set, if you find threshold < 0, then increase scale factor lambda
    # (in rap_defense.py) and train again
    text_list, label_list = process_data(args.constructing_data_path)
    valid_labels = np.array(label_list)
    cls_features, max_features, avg_features = get_embeddings(parallel_model, tokenizer, text_list,  args.batch_size, device)

    np.save('{}/cls_ind_dev_features.npy'.format(output_dir), cls_features)
    np.save('{}/max_ind_dev_features.npy'.format(output_dir), max_features)
    np.save('{}/avg_ind_dev_features.npy'.format(output_dir), avg_features)
    np.save('{}/ind_dev_labels.npy'.format(output_dir), valid_labels)

    # get features of clean samples
    text_list, _ = process_data(args.test_data_path, total_num = args.num_of_samples)
    cls_features, max_features, avg_features = get_embeddings(parallel_model, tokenizer, text_list,\
        args.batch_size, device, target_label=args.protect_label
    )

    np.save('{}/cls_ind_test_clean_features.npy'.format(output_dir), cls_features)
    np.save('{}/max_ind_test_clean_features.npy'.format(output_dir), max_features)
    np.save('{}/avg_ind_test_clean_features.npy'.format(output_dir), avg_features)

    # print(len(clean_output_probs_change_list))
    # get features of poisoned samples
    if args.syntactic_poison_path is None:
        text_list, _ = process_data(args.test_data_path, total_num = args.num_of_samples)
        text_list = data_poison(text_list, backdoor_triggers_list, args.backdoor_trigger_type)
    else:
        text_list, _ = process_data(args.syntactic_poison_path, total_num=args.num_of_samples)

    cls_features, max_features, avg_features = get_embeddings(parallel_model, tokenizer, text_list,\
        args.batch_size, device, target_label=args.protect_label
    )

    np.save('{}/cls_ind_test_poison_features.npy'.format(output_dir), cls_features)
    np.save('{}/max_ind_test_poison_features.npy'.format(output_dir), max_features)
    np.save('{}/avg_ind_test_poison_features.npy'.format(output_dir), avg_features)

 



