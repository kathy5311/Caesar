class Argument:
    def __init__(self, modelname, dropout_rate=0.3):
        self.modelname = modelname
        self.dropout_rate = dropout_rate
        self.nbatch =1
        
        self.maxepoch = 50
        self.datapath = './PDBbind/new_log/new_datass/'
        self.dataf_train = 'train_bin_bal.txt'
        self.dataf_valid = 'valid_bin_bal.txt'
        self.LR = 1.0e-4 #1.0e-4
        self.topk = 32
        self.n_input_feats = 106 #99 (# of nodefeat:3)
        self.channels = 64
        self.nout =26

        self.encoder_args = {'channels': 64, 'num_layers': 2}
        self.decoder_args = {'channels': 64, 'num_layers': 2, 'edge_feat_dim': 3, 'nout': self.nout}

        self.verbose = True
        
args_default = Argument("default")

        
