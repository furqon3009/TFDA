
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x,norm='ortho')
        x_ft2 = torch.fft.fft(x).abs()
    
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1) 
        
        a = out_ft[:, :, :self.modes1].abs() # amplitude
        p = out_ft[:, :, :self.modes1].angle() # phase
        
        return torch.cat([a,p],-1), out_ft

## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

        
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x

    
##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
    
    def forward(self, x):
        predictions = self.logits(x)
        return predictions

class classifier2(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(2*model_output_dim * configs.final_out_channels, configs.num_classes)
        
    
    def forward(self, x):
        predictions = self.logits(x)
        return predictions

class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes   # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length =  configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes*2)   # It doubles because frequency features contain both amplitude and phase
        self.cnn = CNN(configs).to('cuda')  # Time Feature Encoder
        self.cnn_f = CNN(configs).to('cuda')  # Frequency Feature Extractor

        self.projector = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.TSlength_aligned)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.TSlength_aligned)
        )


    def forward(self, xt, xf):
        
        ef, _ = self.cnn_f(xf)
        et, etx = self.cnn(xt)
       
        z_time = self.projector(et)
        z_freq = self.projector_f(ef)
        
        return et, etx, ef, z_time, z_freq 

## Temporal Imputer
class Temporal_Imputer(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        return out

# temporal masking
def masking(x, num_splits=8, num_masked=4):
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask

def get_distances(X, Y, dist_type="cosine"):
        if dist_type == "euclidean":
            distances = torch.cdist(X, Y)
        elif dist_type == "cosine":
            distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
        else:
            raise NotImplementedError(f"{dist_type} distance not implemented.")

        return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(128):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    #Pseudolabels
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(
    features,
    probs,
    banks, num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank, num_neighbors)

    return pred_labels, probs, pred_labels_all, pred_labels_hard

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

@torch.no_grad()
def eval_and_label_dataset(epoch, FE, classifier, banks, test_dataloader, train_dataloader, num_neighbors):
    print("Evaluating Dataset!")
    
    FE.eval()
    classifier.eval()
    logits, indices, gt_labels = [], [], []
    features = []           

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
        # for batch_idx, batch in enumerate(train_dataloader):
            test_inputs, test_targets, test_idxs, test_inputs_f = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[6].cuda()
            # inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
            feats, _, _, _, _ = FE(test_inputs, test_inputs_f)
            # print(f'batch_idx:{batch_idx}')
            # print(f'feats.shape:{feats.shape}')

            logits_cls = classifier(feats)

            features.append(feats)
            gt_labels.append(test_targets)
            logits.append(logits_cls)
            indices.append(test_idxs)  

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384], #hanya diacak urutan indeksnya menggunakan rand_idxs
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }
        
    FE.train()
    classifier.train()
    return banks

