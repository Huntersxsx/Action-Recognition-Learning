class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains UCF-101 dataset
            root_dir = '../UCF-datasets/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '../UCF-datasets/preprocessed'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '../HMDB-datasets/hmdb51'

            output_dir = '../HMDB-datasets/preprocessed'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './pretrained/c3d-pretrained.pth'

