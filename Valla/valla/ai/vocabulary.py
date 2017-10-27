from valla.textmanipulation.textstats import TextStats
import utils.config as cfg

''' Init Vocabulary '''
text_stats = TextStats()
cfg_text_sources = cfg.Config('config.ini').section('TEXT_SOURCES')
source_dir = cfg_text_sources['sources_dir']
vocabulary = list(text_stats.vocabulary_in_files(source_dir))
vocabulary_size = float(len(vocabulary))

print("Vocabulary size: "+str(int(vocabulary_size)))

training_vocabulary = vocabulary[:int(vocabulary_size*(2.0/3.0))]
validation_vocabulary = vocabulary[int(vocabulary_size*(2.0/3.0)):int(vocabulary_size*(5.0/6.0))]
decoding_vocabulary = vocabulary[int(vocabulary_size*(5.0/6.0)):]

print("Training vocabulary size: "+str(len(training_vocabulary)))
print("Validation vocabulary size: "+str(len(validation_vocabulary)))
print("Decoding vocabulary size: "+str(len(decoding_vocabulary)))