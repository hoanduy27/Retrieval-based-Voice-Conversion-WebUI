import logging
import traceback

import numpy as np
from dotenv import load_dotenv

from configs.config import Config
from infer.lib.audio import load_audio, wav2
from infer.modules.vc.modules import  VC ,GuidedVC
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *
import soundfile as sf
import pandas as pd 
from tqdm import tqdm

tqdm.pandas()
logger = logging.getLogger(__name__)

load_dotenv()

config = Config()
config.device = 'cuda:1'
vc = VC(config)
spk_item, _, _, index, _ = vc.get_vc(
    'ESD_0012.pth', 0.33, 0.33
)



if __name__ == "__main__":
    def voice_conversion(in_path, out_path):
        message, (sr, audio) = vc.vc_single(
            0,
            in_path,
            f0_up_key=4,
            f0_file=None,
            f0_method="rmvpe",
            file_index=index["value"],
            file_index2="",
            index_rate=0.75,
            filter_radius=3,
            resample_sr=16000,
            rms_mix_rate=0.25,
            protect=0.33
        )
        # print(message)

        sf.write(out_path, audio, samplerate=sr)


    audio_dir = 'data/'
    syn_dir = 'data/0012_syn/'
    os.makedirs(syn_dir, exist_ok=True)

    df = pd.read_csv('data/sample.csv')

    def single_run(row):
        source_path = os.path.join(audio_dir, row.content_utt)
    
        source_emo = os.path.basename(os.path.dirname(row.content_utt))
        source_name = os.path.splitext(os.path.basename(row.content_utt))[0]

        output_dir = os.path.join(syn_dir, source_emo)
        output_name = f'{source_name}.wav'

        output_path = os.path.join(output_dir, output_name)

        os.makedirs(output_dir, exist_ok=True)
        voice_conversion(source_path, output_path)

    df.progress_apply(
        single_run, axis=1
    )