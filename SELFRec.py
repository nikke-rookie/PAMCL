from data.loader import FileIO
from time import strftime, localtime, time

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config

        print('Reading data and preprocessing...')

        self.training_data = FileIO.load_data_set(config['training.set'])
        self.test_data = FileIO.load_data_set(config['test.set'])

        self.kwargs = {}
        self.kwargs['timestamp'] = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.kwargs['model_name'] = config['model']['name']

        if config.contain('image_modal') and config['image_modal']['fusion']:
            self.kwargs['image_modal'] = config['image_modal']

        if config.contain('text_modal') and config['text_modal']['fusion']:
            self.kwargs['text_modal'] = config['text_modal']

        if config.contain('user_pref') and config['user_pref']['fusion']:
            self.kwargs['user_pref'] = config['user_pref']

    def execute(self):
        import_str = f"from {self.config['model']['name']} import {self.config['model']['name']}"
        exec(import_str)
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        eval(recommender).execute()
