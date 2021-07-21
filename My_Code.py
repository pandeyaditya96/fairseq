import torch
import fairseq

# cp_path = 'C:/Tarun/Pandey Conwin/wav2vec_large.pt'
cp_path = './wav2vec_small.pt'

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()
split_numbers = 1000

for i in range(1000):
    wav_input_16khz = torch.randn(1,15360) # 1000 iterations of 0.96 secs audio 0.96x1000x16000 = 15360000
    print(len(wav_input_16khz[0]))

    z = model.feature_extractor(wav_input_16khz)
    c = model.forward(wav_input_16khz, mask=False, features_only=True)
    print(c)
    print(len(c['features'][0][0]))
