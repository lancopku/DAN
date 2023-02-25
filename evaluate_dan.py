from cmath import log
import torch
import argparse
import numpy as np
import sklearn.covariance
import torch.nn.functional as F
from metrics import get_metrics
from loguru import logger
from torch.nn import CosineSimilarity

# inter-layer pooling
def pooling_features(features, pooling='last', fusion_module=None): # layers, num_samples, hidden_size
    num_layers = features.shape[0]
    if pooling == 'last':
        return features[-1,:,:]
    elif pooling == 'avg':
        return np.mean(features[1:], axis=0)
    elif pooling == 'avg_emb': # including token embeddings
        return np.mean(features, axis=0)
    elif pooling == 'emb':
        return features[0]
    elif pooling == 'first_last':
        return (features[-1] + features[1])/2.0
    elif pooling == 'odd':
        odd_layers= [1 + i for i in range(0, num_layers-1,2)]
        return (np.sum(features[odd_layers],axis=0))/(num_layers/2)
    elif pooling == 'even':
        even_layers= [2 + i for i in range(0, num_layers-1,2)]
        return (np.sum(features[even_layers],axis=0))/(num_layers/2)
    elif pooling == 'last2':
        return (features[-1] + features[-2])/2.0
    elif pooling == 'concat':
        features =  np.transpose(features, (1,0,2)) # num_samples, layers, hidden_size
        return features.reshape(features.shape[0],-1) # num_samples, layers*hidden_size
    elif type(pooling) == int or (type(pooling) == str and pooling.isdigit()):
        pooling = int(pooling)
        return features[pooling]
    elif ',' in pooling or type(pooling) == list:
        layers = pooling
        if type(pooling) == str:
            layers = list([int(l) for l in pooling.split(',')])
        return np.mean(features[layers], axis=0)
    else:
        raise NotImplementedError

def sample_estimator(features, labels):
    labels = labels.reshape(-1)
    num_classes = np.unique(labels).shape[0]
    print(num_classes)
    #group_lasso = EmpiricalCovariance(assume_centered=False)
    #group_lasso =  MinCovDet(assume_centered=False, random_state=42, support_fraction=1.0)
    group_lasso = sklearn.covariance.ShrunkCovariance()
    sample_class_mean = []
    #class_covs = []
    for c in range(num_classes):
        current_class_mean = np.mean(features[labels==c,:], axis=0)
        sample_class_mean.append(current_class_mean)
        #cov_now = np.cov((features[labels == c]-(current_class_mean.reshape([1,-1]))).T)
        #class_covs.append(cov_now)
    #precision = np.linalg.inv(np.mean(np.stack(class_covs,axis=0),axis=0))
    #print(precision.shape)
    #  
    X  = [features[labels==c,:] - sample_class_mean[c]  for c in range(num_classes)]
    X = np.concatenate(X, axis=0)
    group_lasso.fit(X)
    precision = group_lasso.precision_

    return sample_class_mean, precision

def get_distance_score(class_mean, precision, features, measure='maha'):
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        if measure == 'maha':
            score = -1.0*torch.mm(torch.mm(centered_features, precision), centered_features.t()).diag()
        elif measure == 'euclid':
            score = -1.0*torch.mm(centered_features, centered_features.t()).diag()
        elif measure == 'cosine':
            score = torch.tensor([CosineSimilarity()(features[i].reshape(1,-1), class_mean[c].reshape(1,-1)) for i in range(num_samples)])
        scores.append(score.reshape(-1,1))
    scores = torch.cat(scores, dim=1) # num_samples, num_classes
    print(scores.shape)
    scores,_ = torch.max(scores, dim=1) # num_samples
    #scores = scores[:,1]
    scores = scores.cpu().numpy()
    return scores       



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--ood_method', default='base', type=str)
    parser.add_argument('--ood_datasets', default='20news,wmt16,multi30k,rte,snli',
                        type=str, required=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        required=False, help='batch size')
    parser.add_argument('--distance_metric', type=str, default='maha',
                        help='distance metric')
    parser.add_argument('--token_pooling', type=str, default='avg',
                        help='token pooling way', choices = ['cls','avg', 'max'])
    parser.add_argument('--layer_pooling', type=str, default='last')
    parser.add_argument('--output_image', type=str, default=None)
    parser.add_argument('--input_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
                        type=str, required=False, help='save directory')
    parser.add_argument('--log_file', type=str, default='./log/inter_results.log')
    parser.add_argument('--score_ensemble', action='store_true')
    parser.add_argument('--std', action='store_true')
    parser.add_argument('--agg', type=str, default='mean', choices=['mean','min'])
    parser.add_argument('--layer_analysis', action='store_true')
    parser.add_argument('--model_path', default='./log/embeddings/roberta-base/sst-2/seed13/avg_lstm_model.pt')
    parser.add_argument('--valid_size', type=int, default=-1)
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())


    input_dir = args.input_dir
    token_pooling = args.token_pooling
    layer_pooling = args.layer_pooling

    logger.info(input_dir)
    logger.info(token_pooling)
    logger.info(layer_pooling)

    if args.layer_analysis:
        ood_full_features_list = []
        for ood_dataset in args.ood_datasets.split(','):
            ood_features = np.load('{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
            ood_full_features_list.append(ood_features)
        ind_test_full_features = np.load('{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
        ind_train_full_features = np.load('{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
        ind_train_labels = np.load('{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))
        num_layers = ind_train_full_features.shape[0]-1
        best_aurocs = []
        best_combination = None
        for i in range(1, num_layers+1):
            best_auroc = 0 
            # Search for the best combination
            from itertools import combinations
            for choice in combinations([j for j in range(1, num_layers+1)], i):
                ind_train_features = pooling_features(ind_train_full_features, list(choice))
                sample_class_mean, precision = sample_estimator(ind_train_features, ind_train_labels)
                ind_test_features = pooling_features(ind_test_full_features, list(choice))
                ind_scores = get_distance_score(sample_class_mean, precision,\
                    ind_test_features, measure=args.distance_metric)
                aurocs = []
                for ood_full_features in ood_full_features_list:
                    ood_test_features = pooling_features(ood_full_features, list(choice))
                    ood_scores = get_distance_score(sample_class_mean, precision, ood_test_features, measure=args.distance_metric)
                    auroc = get_metrics(ind_scores, ood_scores)['AUROC']
                    aurocs.append(auroc)
                mean_auroc = sum(aurocs)/len(aurocs)
                if best_auroc < mean_auroc:
                    best_auroc = mean_auroc
                    if len(best_aurocs) == 0 or best_auroc > max(best_aurocs):
                        best_combination = list(choice)
                #best_auroc = max(best_auroc, np.mean(aurocs))
            best_aurocs.append(best_auroc)
        print(best_aurocs)
        print(best_combination)
        return



    if args.score_ensemble:
        ind_dev_features = np.load('{}/{}_ind_dev_features.npy'.format(input_dir, token_pooling))
        ind_dev_labels = np.load('{}/ind_dev_labels.npy'.format(input_dir))
        num_layers = ind_dev_features.shape[0] - 1
        
        indices = np.arange(ind_dev_features.shape[1])
        np.random.seed(42)
        np.random.shuffle(indices)
        valid_size = int(0.2*ind_dev_features.shape[1])
        ind_dev_features_train, ind_dev_features_valid = ind_dev_features[:,indices[:-valid_size]], ind_dev_features[:,indices[-valid_size:]]
        ind_dev_labels_train, ind_dev_labels_valid = ind_dev_labels[indices[:-valid_size]], ind_dev_labels[indices[-valid_size:]] 

        clean_test_features = np.load('{}/{}_ind_test_clean_features.npy'.format(input_dir, token_pooling))
        poison_test_features = np.load('{}/{}_ind_test_poison_features.npy'.format(input_dir, token_pooling))

        clean_scores_list = []
        poison_scores_list = []
        valid_scores_list = []
        for layer in range(1, num_layers+1):
            print("layer {}".format(layer))
            ind_train_features = ind_dev_features_train[layer]
            sample_class_mean, precision = sample_estimator(ind_train_features, ind_dev_labels_train)
            valid_scores = -1*get_distance_score(sample_class_mean, precision, ind_dev_features_valid[layer], measure=args.distance_metric)
            clean_scores = -1*get_distance_score(sample_class_mean, precision, clean_test_features[layer], measure=args.distance_metric)
            poison_scores = -1*get_distance_score(sample_class_mean, precision, poison_test_features[layer], measure=args.distance_metric)
            if args.std:
                mean = np.mean(valid_scores)
                std = np.std(valid_scores)
                valid_scores = (valid_scores - mean)/std
                clean_scores = (clean_scores - mean)/std
                poison_scores = (poison_scores - mean)/std
            clean_scores_list.append(-1*clean_scores)
            poison_scores_list.append(-1*poison_scores)
            valid_scores_list.append(-1*valid_scores) 
        if args.agg == 'mean':
            clean_scores = np.mean(clean_scores_list, axis=0)
            valid_scores = np.mean(valid_scores_list, axis=0)
            poison_scores = np.mean(poison_scores_list, axis=0)
        elif args.agg == 'min':
            clean_scores = np.min(clean_scores_list, axis=0)
            valid_scores = np.min(valid_scores_list, axis=0)
            poison_scores = np.min(poison_scores_list, axis=0)
        
        
        metrics = get_metrics(clean_scores, poison_scores, valid_scores)
        logger.info('AUROC :{:.2f}%'.format(metrics['AUROC']*100))
        for  FRR in [0.5,1,3,5,10]:
            logger.info('valid FRR={}, FRR :{:.2f}%, FAR :{:.2f}%'.format(FRR, metrics["FRR_backdoor_FRR_{}".format(FRR)]*100, metrics["FAR_backdoor_FRR_{}".format(FRR)]*100))

        return 



    ind_dev_features = np.load('{}/{}_ind_dev_features.npy'.format(input_dir, token_pooling))
    #ind_train_features = torch.load('{}/{}_ind_train_features.pt'.format(input_dir, token_pooling))
    ind_dev_labels = np.load('{}/ind_dev_labels.npy'.format(input_dir))

    
    ind_dev_features = pooling_features(ind_dev_features, layer_pooling)
    if args.valid_size != -1:
        indices = np.arange(len(ind_dev_features))
        np.random.seed(42)
        np.random.shuffle(indices)
        ind_dev_features = ind_dev_features[indices[:args.valid_size]]
        ind_dev_labels = ind_dev_labels[indices[:args.valid_size]]
    indices = np.arange(len(ind_dev_features))
    np.random.seed(42)
    np.random.shuffle(indices)
    valid_size = int(0.2*len(ind_dev_features))
    ind_dev_features_train, ind_dev_features_valid = ind_dev_features[indices[:-valid_size]], ind_dev_features[indices[-valid_size:]]
    ind_dev_labels_train, ind_dev_labels_valid = ind_dev_labels[indices[:-valid_size]], ind_dev_labels[indices[-valid_size:]] 
    sample_class_mean, precision = sample_estimator(ind_dev_features_train, ind_dev_labels_train)
    
    clean_test_features = np.load('{}/{}_ind_test_clean_features.npy'.format(input_dir, token_pooling))
    clean_test_features = pooling_features(clean_test_features, layer_pooling)
    clean_scores = get_distance_score(sample_class_mean, precision, clean_test_features, args.distance_metric)

    poison_test_features = np.load('{}/{}_ind_test_poison_features.npy'.format(input_dir, token_pooling))
    poison_test_features = pooling_features(poison_test_features, layer_pooling)
    poison_scores = get_distance_score(sample_class_mean, precision, poison_test_features, args.distance_metric)
    
    valid_scores = get_distance_score(sample_class_mean, precision, ind_dev_features_valid, args.distance_metric)
    metrics = get_metrics(clean_scores, poison_scores, valid_scores)
    logger.info('AUROC :{:.2f}%'.format(metrics['AUROC']*100))
    for  FRR in [0.5,1,3,5,10]:
        logger.info('valid FRR={}, FRR :{:.2f}%, FAR :{:.2f}%'.format(FRR, metrics["FRR_backdoor_FRR_{}".format(FRR)]*100, metrics["FAR_backdoor_FRR_{}".format(FRR)]*100))




if __name__ == '__main__':
    main()
