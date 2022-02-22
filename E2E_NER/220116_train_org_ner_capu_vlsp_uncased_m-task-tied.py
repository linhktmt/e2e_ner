import os, sys
from datetime import datetime
from flair.data import Corpus
from flair.datasets import ColumnCorpus
# from m_sequence_tagger_model_211108 import SequenceTagger
import m_embeddings
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
import flair

Model_lib = 'm_sequence_tagger_model_m-task-tied_220115' # TODO: Capu lamda = 1000
import importlib
model = importlib.import_module(Model_lib)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_corpus():
    # define columns
    columns = {0: 'text', 1: 'ner', 2: 'capu'}

    # this is the folder in which train, test and dev files reside
    data_folder = './data/vlsp_org_uncased_cleaned_full_tagged'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  dev_file='dev.txt',
                                  test_file='test.txt')
    print(corpus)
    return corpus


if __name__ == "__main__":
    time = datetime.now().strftime("%y%m%d-%H%M%S")
    save_dir = os.path.join("./save", 'vlsp_ner_capu_org_uncased_cleaned_m-task-tied_{}/'.format(time))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    script_fname =  sys.argv[0]

    # from m_sequence_tagger_model_211108 import SequenceTagger

    import shutil
    shutil.copy(script_fname, save_dir)  # Store code to save dir
    shutil.copy(Model_lib + '.py', save_dir)



    # 1. get the corpus
    dataset = load_corpus()
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary, tag_capu_dictionary = dataset.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)
    print(tag_capu_dictionary.idx2item)
    # 4. initialize embeddings
    m_embeddings.get_trained_embedding('./model-bin/vibert/')

    # 5. initialize sequence tagger
    tagger = model.SequenceTagger(hidden_size=512,
                                            rnn_layers=4,
                                            rnn_input_dim=m_embeddings.get_trained_embedding().embedding_length,
                                            embedding_length=m_embeddings.get_trained_embedding().embedding_length,
                                            tag_dictionary=tag_dictionary,
                                            tag_capu_dictionary=tag_capu_dictionary,
                                            capu_dropout = 0.0,
                                            tag_type=tag_type,
                                            use_crf=True,
                                            rnn_type="GRU"
                                            )

    # 6. initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, dataset)
    #
    # # 7. start training
    trainer.train(save_dir,
                  learning_rate=0.1,
                  mini_batch_size=64,
                  max_epochs=200,
                  anneal_factor=0.25,
                  min_learning_rate=0.000001,
                  embeddings_storage_mode='cpu',
                  )

    # 8. plot weight traces (optional)
    plotter = Plotter()
    plotter.plot_weights('{}/weights.txt'.format(save_dir))
