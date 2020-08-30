import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
"""
Diese Datei ist dafür da, um unsere Raw Datein zu "säubern". 
Leerstellen/nichtsaussagende Stellen in den Datein sollen rausgefiltert werden. 
2 Spurige Audio Datein sollen in Mono formatiert werden. 
Windowing soll durchgeführt werden, in dem immer nur ein bis maximal 2 Herzschläge pro gesäuberte Datein vorhanden sind. (Außerdem hat man dann mehr Datein zum üben) 

"""

def envelope(y, rate, threshold):
    """
    Diese Funktion ist dafür dar, um Leerstellen rauszufiltern. 
    """
    mask = []
    # Damit wir nicht die Datei zerstören, führen wir ein rolling durch und den absoluten Betrag. Denn durch den threshhold werden wir alles was unter 100 Hertz geht, rausgefiltert.
    # Durch den abosulten Wert, verringern wir die Chance, dass es solche stellen gibt. (Man muss sich hierfür einen Graphen vorstellen von Audio Datein. Die gehen auch ins Negative)
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):

    rate, wav = wavfile.read(path)
    wav = wav.astype(np.float32, order='F')
    try:
        tmp = wav.shape[1]
        wav = to_mono(wav.T)
    except:
        pass
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # gesäuberte Datei ist weniger als ein sample
            # Blöcke mit 0 in delta_sample Größe anpassen
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int32)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # Audio durchlaufen und jedes delta_sample speichern
            # den Endton verwerfen, wenn er zu kurz ist
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    #Wo befinden sich die Raw Datein? 
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='directory of audio files in total duration')
    # WO sollen die sauberen Datein gespeichert werden? 
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='directory to put audio files split by delta_time')
    #In was für Länge wollen wir die Datei aufteilen? ein Herzschlag braucht ca. 0.2 sekunden. 
    parser.add_argument('--delta_time', '-dt', type=float, default=0.25,
                        help='time in seconds to sample audio')
    #AUf welchen Wert wollen wir die Datein downsamplen? Der Herzschlag sollte sich bei 200 befinden. 
    parser.add_argument('--sr', type=int, default=5000,
                        help='rate to downsample audio')
    #Welche Datei soll al Test angezeigt werden? 
    parser.add_argument('--fn', type=str, default='a100.wav',
                        help='file to plot over time to check magnitude')
    #Ab welcher Frequenz soll rausgeschnitten werden, um Rauschen rauszufiltern oder Leerstellen?
    parser.add_argument('--threshold', type=str, default=50,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    test_threshold(args)
    split_wavs(args)
