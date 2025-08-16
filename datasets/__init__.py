
#build dataset according to given 'dataset_file'

def build_dataset(args):
    if args["dataset_file"] == 'SHA':
        from datasets.SHA.loading_data import loading_data
        return loading_data

    return None



