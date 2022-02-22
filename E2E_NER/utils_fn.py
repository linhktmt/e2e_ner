import os
import shutil
import numpy as np
import pickle
import logging
import random
import json
import torch
# import matplotlib.pyplot as plt

# from sklearn.utils import shuffle
# from sklearn.preprocessing.data import QuantileTransformer, MinMaxScaler

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benmark = False
        torch.backends.cudnn.deterministic = True


def to_onehot(y_in):
    num = np.unique(y_in, axis=0)
    num = num.shape[0]
    y_out = np.eye(num)[y_in]
    return y_out

def read_cv(x, y, n_fold):
    dev_ratio = 0.1
    # tr_idx = int(len(labels) * tr_ratio)
    data = {}
    start_index = (n_fold - 1) * int(dev_ratio * len(x))
    end_index = n_fold * int(dev_ratio * len(x))
    ex1 = x[0:start_index]
    ex2 = x[end_index:len(x)]
    ex_train = np.concatenate((ex1,ex2),axis=0)
    y_train = np.concatenate((y[0:start_index] , y[end_index:len(x)]),axis=0)
    dev_idx = len(ex_train) // 9 * 8

    data["train_x"] = ex_train[:dev_idx]
    data["train_y"] = y_train[:dev_idx]
    data["dev_x"], data["dev_y"] = ex_train[dev_idx:len(ex_train)], y_train[dev_idx:len(ex_train)]

    data["test_x"] = x[start_index:end_index]
    data["test_y"] = y[start_index:end_index]
    return data



def save_model(model, params):
    path = "saved_models/{}_{}_{}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {path}!".format(path))


def load_model(params):
    path = "saved_models/{}_{}_{}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])

    try:
        model = pickle.load(open(path, "rb"))
        print("Model in {} loaded successfully!".format(path))

        return model
    except:
        print("No available model such as {}.".format(path))
        exit()

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger, handler

def log_params(logger, params):
     logger.info("=" * 20 + "PARAMETERS" + "=" * 20)
     for keys, values in params.items():
          st = "{}: {}".format(keys, values)
          logger.info(st)     
     logger.info("=" * 40)
     return True


def log_args(logger, args):
    print("=" * 20 + "PARAMETERS" + "=" * 20)
    logger.info("=" * 20 + "PARAMETERS" + "=" * 20)
    for arg in vars(args):
        logger.info('{} = {}'.format(arg, getattr(args, arg)))
        print('{} = {}'.format(arg, getattr(args, arg)))
    logger.info("=" * 40)
    print("=" * 40)
    return True

def update_params(params):
    if params["MODEL"] == "multichannel":
        params["IN_CHANNEL"] = 2
    else: params["IN_CHANNEL"] = 1
    method_name = "read_cv_{}".format(params["DATASET"])
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(method_name)
    if not method:
         raise NotImplementedError("Method %s not implemented" % method_name)
    data_init = method(0)
#    data_init = getattr(utils, "read_cv_{}".format(params["DATASET"]))(0)
             
    data_init["vocab"] = sorted(list(set([w for sent in data_init["x"] for w in sent])))
    data_init["classes"] = sorted(list(set(data_init["y"])))
    data_init["word_to_idx"] = {w: i for i, w in enumerate(data_init["vocab"])}
    data_init["idx_to_word"] = {i: w for i, w in enumerate(data_init["vocab"])}

    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        w2v_path = "/media/ubuntu/Data01/Linh/py_code/_text_classification/GoogleNews-vectors-negative300.bin"
        word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
        wv_matrix = []
        for i in range(len(data_init["vocab"])):
            word = data_init["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        del word_vectors
        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
        del wv_matrix

    params["MAX_SENT_LEN"]= max([len(sent) for sent in data_init["x"]])
    params["VOCAB_SIZE"] = len(data_init["vocab"])
    params["CLASS_SIZE"] = len(data_init["classes"])
    return params, data_init


###################################
# from torch.nn.modules.module import _addindent
# import torch
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing testable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr


def lr_round(num, num_digit=1):
    from itertools import groupby

    if num>=10:
        rnum = round(num,0)
    elif num>=0.1:
        rnum = round(num, 1)
    else:
        nstr = '{:.9f}'.format(num)
        groups = groupby(nstr)
        result = [(label, sum(1 for _ in group)) for label, group in groups]
        dplace = result[2][1] + num_digit
        rnum = round(num, dplace)

    return rnum



def display_time(seconds, granularity=2):
    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

# def plot_learning_curve(x, scores, figure_file):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 scores')
#     plt.savefig(figure_file)
#
# def plotLearning(scores, filename, x=None, window=5):
#     import matplotlib.pyplot as plt
#     N = len(scores)
#     running_avg = np.empty(N)
#     for t in range(N):
#         running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
#     if x is None:
#         x = [i for i in range(N)]
#     fig = plt.figure()
#     plt.ylabel('Score')
#     plt.xlabel('Running average of previous 100 scores')
#     plt.plot(x, running_avg)
#     plt.savefig(filename)
#     plt.close(fig)

def match_ne_string(s1, s2):
    from fuzzywuzzy import fuzz

    s1l = s1
    s2l = s2
    n_num = len(s1l)
    sum_num = len(s2l)
    num_sets = []
    num_set = np.zeros(n_num).astype(int)

    def get_num(num, n):  # find all possible sub-index for separate s2 to match with s1
        # global n_num, sum_num, num_set, num_sets
        if n == 1:
            num_set[n_num - n] = num
            num_sets.append(np.copy(num_set))
            # print(num_sets)
            return num
        else:
            for i in range(1,num):
                num_set[n_num-n]=i
                num_next = get_num(num-i,n-1)

    l_dists = []
    ne_pairs = []
    if n_num==1:
        for i in range(len(s2l)):
            ne_pairs.append([[s1l[0],[s2l[i]]]])
            l_dists.append(fuzz.ratio(s1l[0],s2l[i]))
        max_idx = l_dists.index(max(l_dists))
    else:
        nums = get_num(sum_num,n_num)
        # l_dists = []
        # ne_pairs = []
        for index in num_sets:
            l_dist = []
            ne_pair = []
            curr_ind = 0
            for p in range(len(s1l)):
                # print(s1l[p], s2l[curr_ind:curr_ind+index[p]])
                ne_pair.append([s1l[p], s2l[curr_ind:curr_ind+index[p]]])
                fz = fuzz.ratio(s1l[p], ''.join(s2l[curr_ind:curr_ind+index[p]]))
                curr_ind += index[p]
                l_dist.append(fz)
            l_dist.append(sum(l_dist))
            l_dists.append(l_dist)
            ne_pairs.append(ne_pair)

        sum_l_dists = [ld[-1] for ld in l_dists]
        max_idx = sum_l_dists.index(max(sum_l_dists))


    return ne_pairs[max_idx], l_dists[max_idx]

def tag_align(err_del_list1, err_ins_list1, ner_ref, tag_idx):
    tag_list = []
    # tag_idx = 0
    # if len(err_ins_list1) > len(err_del_list1):  # insert err -> tag = 'O'
    #     while len(err_ins_list1) > len(err_del_list1):
    #         err_del_list1.append([])
    #
    # elif len(err_ins_list1) < len(err_del_list1):  # delete err -> do nothing
    #     while len(err_ins_list1) < len(err_del_list1):
    #         err_ins_list1.append([])

    while (len(err_del_list1) > 0) or (len(err_ins_list1) > 0):
        l_del_words = len(err_del_list1[0])
        l_ins_words = len(err_ins_list1[0])
        long_tag = False
        swapped = False
        if l_ins_words == 0:  # delete err -> do nothing
            pass
        elif l_del_words == 0:  # insert err -> tag = 'tag_idxO'
            for ins_idx in range(l_ins_words):
                tag_list.append('O')
                # tag_idx += 1
        elif l_del_words == 1:
            for des_idx in range(len(err_ins_list1[0])):
                adding_ne = ner_ref['output'][tag_idx]
                if des_idx >= 1 and adding_ne[0] == 'B':
                    adding_ne = adding_ne.replace('B', 'I', 1)
                tag_list.append(adding_ne)
            tag_idx += 1

        elif l_del_words > l_ins_words:
            src_words, des_words = err_ins_list1[0], err_del_list1[0]  # src_words need shorter than des_words
            swapped = True
            long_tag = True
        else:
            src_words, des_words = err_del_list1[0], err_ins_list1[0]
            long_tag = True

        if long_tag:
            pairs, _ = match_ne_string(src_words, des_words)
            if swapped:
                for p_idx in range(len(pairs)):
                    word_ner_idx = ner_ref['input'].index(pairs[p_idx][1][0])
                    # ne_word = ner_ref['output'][word_ner_idx]
                    adding_ne = ner_ref['output'][word_ner_idx]
                    tag_list.append(adding_ne)
                    # for des_idx in range(len(pair[1])):
                tag_idx += l_del_words
            else:
                for pair in pairs:
                    for des_idx in range(len(pair[1])):
                        adding_ne = ner_ref['output'][tag_idx]
                        if des_idx >= 1 and adding_ne[0] == 'B':
                            adding_ne = adding_ne.replace('B', 'I', 1)
                        tag_list.append(adding_ne)
                    tag_idx += 1
        del (err_del_list1[0])
        del (err_ins_list1[0])

    return tag_list, tag_idx

def tag_capu_align(err_del_list1, err_ins_list1, ner_ref, tag_idx):
    tag_list = []

    while (len(err_del_list1) > 0) or (len(err_ins_list1) > 0):
        l_del_words = len(err_del_list1[0])
        l_ins_words = len(err_ins_list1[0])
        long_tag = False
        swapped = False
        if l_ins_words == 0:  # delete err -> do nothing
            pass
        elif l_del_words == 0:  # insert err -> tag = 'tag_idxO'
            for ins_idx in range(l_ins_words):
                tag_list.append('L$')
                # tag_idx += 1
        elif l_del_words == 1:
            added_nes = []
            des_capu = err_ins_list1[0]
            for des_idx in range(len(des_capu)-1, -1, -1):
                adding_ne = ner_ref['output'][tag_idx]
                capu = adding_ne[1]
                if des_idx < len(des_capu)-1 and capu in ['.',',','?','!']: # find from end of ins list
                    adding_ne = adding_ne.replace(capu, '$', 1)
                    added_nes.insert(0,adding_ne)
                else:
                    added_nes.insert(0, adding_ne)
            tag_list.extend(added_nes)
            tag_idx += 1

        elif l_del_words > l_ins_words:
            src_words, des_words = err_ins_list1[0], err_del_list1[0]  # src_words need shorter than des_words
            swapped = True
            long_tag = True
        else:
            src_words, des_words = err_del_list1[0], err_ins_list1[0]
            long_tag = True

        if long_tag:
            pairs, _ = match_ne_string(src_words, des_words)
            if swapped:
                for p_idx in range(len(pairs)):
                    word_ner_idx = ner_ref['input'].index(pairs[p_idx][1][0])
                    # ne_word = ner_ref['output'][word_ner_idx]
                    adding_ne = ner_ref['output'][word_ner_idx]
                    tag_list.append(adding_ne)
                    # for des_idx in range(len(pair[1])):
                tag_idx += l_del_words
            else:
                for pair in pairs:
                    added_nes = []
                    # for des_idx in range(len(pair[1])-1, -1, -1): #TODO: not err_ins_list
                    des_capu = pair[1]
                    for des_idx in range(len(des_capu) - 1, -1, -1):
                        adding_ne = ner_ref['output'][tag_idx]
                        capu = adding_ne[1]
                        if des_idx < len(des_capu) - 1 and capu in ['.', ',', '?','!']:  # find from end of ins list
                            adding_ne = adding_ne.replace(capu, '$', 1)
                            added_nes.insert(0, adding_ne)
                        else:
                            added_nes.insert(0, adding_ne)
                    tag_list.extend(added_nes)
                    tag_idx += 1
        del (err_del_list1[0])
        del (err_ins_list1[0])

    return tag_list, tag_idx

def clean_text(s, lower=False):
    import re
    if lower:
        s = s.lower()
    # s = "string. With. Punctuation?"
    s = re.sub(r'[^\w\s]', '', s)
    return s

def save_file_list_to_text(file_path_write, lines_write):
    with open(file_path_write, 'w', encoding='utf-8') as file:
        for line in lines_write:
            file.write('{}\n'.format(line))