import torch
import fairseq

cp_path = './wav2vec_small_960h.pt'
temp_cp = torch.load(cp_path)

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([temp_cp])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
