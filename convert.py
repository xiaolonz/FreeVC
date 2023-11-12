import glob
import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file")
    # parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--targetfile", type=str, default="/data4/xiaolonz/Vctk/vctk-16k/p243/p243_394.wav", help="target "
                                                                                                              "wav")
    parser.add_argument("--srcpath", type=str, default="/data4/xiaolonz/Vctk/vctk_16k_all", help="src dir")
    parser.add_argument("--outdir", type=str, default="/data4/xiaolonz/Vctk/vctk_16k_out", help="out dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # print("Processing text...")
    # titles, srcs, tgts = [], [], []
    # with open(args.txtpath, "r") as f:
    #     for rawline in f.readlines():
    #         title, src, tgt = rawline.strip().split("|")
    #         titles.append(title)
    #         srcs.append(src)
    #         tgts.append(tgt)

    # tgt
    tgt = args.targetfile
    wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    if hps.model.use_spk:
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    else:
        wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
        mel_tgt = mel_spectrogram_torch(
            wav_tgt,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

    print("Synthesizing...")
    with torch.no_grad():
        for srcfile in glob.glob(os.path.join(args.srcpath, '*.wav')):
            # src
            basename = os.path.basename(srcfile)
            src, extension = os.path.splitext(basename)
            wav_src1, _ = librosa.load(srcfile, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src1).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)
            
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+srcfile)), hps.data.sampling_rate, audio)
            else:
                delay = 160*5
                write(os.path.join(args.outdir, f"{src}_converted.wav"), hps.data.sampling_rate,
                      audio[3200+delay:-3200])
                len_audio = len(audio[3200+delay:-3200])
                write(os.path.join(args.outdir, f"{src}_origin.wav"), hps.data.sampling_rate,
                      wav_src1[3200:3200+len_audio])


            
