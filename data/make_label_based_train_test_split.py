import sys
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
import json
from pathlib import Path
import datetime
from sklearn.model_selection import StratifiedKFold
from base.datasets.table_reader import TableReader
from base.options.train_options import TrainOptions
from base.utils import utils


def main():
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--dataset_name') or arg.startswith('--task'):
            sys.argv.pop(i)
    sys.argv.append('--task=phenotype')  # FIXME no need for this, as parser.parse_args can take partial string input
    sys.argv.append('--dataset_name=tcga')
    train_options = TrainOptions()
    train_options.parser.add_argument('--num_splits', default=3, type=int, help="Determine num of splits")
    train_options.parser.add_argument('--target_genes', default='PTEN', type=str, help="Target gene(s)")
    opt = train_options.parse()
    # read metadata
    field_names = opt.data_fields.split(',')
    datatypes = opt.field_datatypes.split(',')
    sample = TableReader(field_names, datatypes)
    cna = TableReader(field_names, datatypes)
    data = None
    wsi_replacements = {
        'FALSE': False,
        'TRUE': True,
        'released': True
    }
    sample.read_singleentry_data(opt.wsi_tablefile, replace_dict=wsi_replacements)
    sample.index_data(index=opt.sample_index)
    cna.read_matrix_data(opt.cna_tablefile, yfield='Hugo_Symbol', xfield=(0, 2))
    cna.index_data(index='y')
    sample.data.query("is_ffpe == True", inplace=True)  # remove all slides that are not FFPE
    # split into subsets
    target_gene = opt.target_genes.split(',')[0]  # TODO make this work for any number of target genes
    cna_score = cna.data.loc[:, target_gene]
    loss = cna.data.loc[cna_score < 0, target_gene]
    # make splits:
    skf = StratifiedKFold(n_splits=opt.num_splits)
    splits = skf.split(cna.data.index.values, (cna.data.loc[:, target_gene] < 0).values)
    save_path = Path(opt.data_dir)/'CVsplits'
    save_path.mkdir(exist_ok=True)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_ids = cna.data.index[train_idx].values  # get slide ids using indices
        test_ids = cna.data.index[test_idx].values
        to_dump = {
                'date': datetime.datetime.now().__str__(),
                'stratification': opt.target_genes,
                'num_splits': opt.num_splits,
                'split_num': i,
                'train': list(train_ids),  # ndarray cannot be serialized to JSON in python 3.7
                'test': list(test_ids),
                'class_counts': {
                    'train': {},
                    'test': {}
                }
            }
        for value in cna_score.unique():
            value = value.item()  # convert to native python type for serialization
            to_dump['class_counts']['train'][value] = (cna_score[train_ids] == value).sum().item()
            to_dump['class_counts']['test'][value] = (cna_score[test_ids] == value).sum().item()
        with open(save_path/f'split{i}.json', 'w') as split_json:
            json.dump(to_dump, split_json)


if __name__ == '__main__':
    main()
