from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence,text_to_sequence2, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging
from winsound import PlaySound
# 加入百度翻译日语
from translateBaidu import translate_baidu

chinese_model_path = ".\model\CN\model.pth"
chinese_config_path = ".\model\CN\config.json"
japanese_model_path = ".\model\H_excluded.pth"
japanese_config_path = ".\model\config.json"
extern_model_path = ".\model\extern\yazawa_Nico.pth"
extern_config_path = ".\model\extern\yazawa_config.json"
genshin_model_path = ".\model\genshin\genshin.pth"
genshin_config_path = ".\model\genshin\genshin.json"

####################################
#CHATGPT INITIALIZE
from pyChatGPT import ChatGPT
import json

modelmessage = """ID      Output Language
0       Chinese
1       Japanese
2       genshin
"""

idmessage_cn = """ID      Speaker
0       綾地寧々
1       在原七海
2       小茸
3       唐乐吟
"""

idmessage_jp = """ID      Speaker
0       綾地寧々
1       因幡めぐる
2       朝武芳乃
3       常陸茉子
4       ムラサメ
5       鞍馬小春
6       在原七海
"""
idmessage_gs = """ID      Speaker
0	派蒙 1	凯亚 2	安柏 3	丽莎 4  琴 5  香菱 6	枫原万叶 7  迪卢克 8  温迪 9  可莉
10	早柚 11	托马 12	芭芭拉 13  优菈 14  云堇 15  钟离 16  魈 17	凝光 18	雷电将军 19	北斗
20	甘雨 21	七七 22	刻晴 23	神里绫华 24	戴因斯雷布 25	雷泽 26	神里绫人 27	罗莎莉亚 28	阿贝多 29  八重神子
30	宵宫 31	荒泷一斗 32	九条裟罗 33	夜兰 34	珊瑚宫心海 35	五郎 36	散兵 37	女士 38	达达利亚 39  莫娜
40	班尼特 41  申鹤 42  行秋 43  烟绯 44  久岐忍 45  辛焱 46  砂糖 47  胡桃 48  重云 49  菲谢尔 
50	诺艾尔 51  迪奥娜 52  鹿野院平藏
"""

def get_input():
    # prompt for input
    print("You:")
    user_input = input()
    return user_input

def get_input_jp():
    # prompt for input
    print("You:")
    user_input = input() +" 使用日本语"
    return user_input

def get_token():
    token = input("Copy your token from ChatGPT and press Enter \n")
    return token

      
################################################


logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_text2(text, hps):
    text_norm = text_to_sequence2(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    text_norm = LongTensor(text_norm)
    return text_norm
def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text



def generateSound(inputString, id, model_id):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    #model = input('0: Chinese')
    #config = input('Path of a config file: ')
    if model_id == 0:
        model = chinese_model_path
        config = chinese_config_path
    elif model_id == 1:
        model = japanese_model_path
        config = japanese_config_path
    elif model_id == 2:
        model = genshin_model_path
        config = genshin_config_path

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        if not emotion_embedding:
            #while True:
            if(1 == 1):
                choice = 't'
                if choice == 't':
                    text = inputString
                    if text == '[ADVANCED]':
                        text = "我不会说"

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    if model_id == 2:
                        stn_tst = get_text2(text, hps_ms)
                    else:
                        stn_tst = get_text(text, hps_ms, cleaned=cleaned)


                    speaker_id = id 
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')


def generateSound_extern(inputString, id, model_id):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    # model = input('0: Chinese')
    # config = input('Path of a config file: ')
    if model_id == 0:
        model = chinese_model_path
        config = chinese_config_path
    elif model_id == 1:
        model = japanese_model_path
        config = japanese_config_path

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        if not emotion_embedding:
            # while True:
            if (1 == 1):
                choice = 't'
                if choice == 't':
                    text = inputString
                    if text == '[ADVANCED]':
                        text = "我不会说"

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    speaker_id = id
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                            0, 0].data.cpu().float().numpy()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')

if __name__ == "__main__":
    session_token = get_token()
    api = ChatGPT(session_token)
    print(modelmessage)
    model_id = int(input('选择回复语言: '))
    if model_id == 0:
        print("\n" + idmessage_cn)
        id = get_speaker_id('选择角色: ')
    elif model_id == 1:
        print("\n" + idmessage_jp)
        id = get_speaker_id('选择角色: ')
    elif model_id == 2:
        print("\n" + idmessage_gs)
        id = get_speaker_id('选择角色: ')
    print()
    while True:

        if model_id == 0:
            resp = api.send_message(get_input())
            if(resp == "quit()"):
                break
            answer = resp["message"].replace('\n','')
            print("ChatGPT:")
            print(answer)
            generateSound("[ZH]"+answer+"[ZH]", id, model_id)
            PlaySound(r'.\output.wav', flags=1)

            pass
        elif model_id == 1:
            resp = api.send_message(get_input_jp())
            if(resp == "quit()"):
                break
            answer = resp["message"].replace('\n','')
            print("ChatGPT:")
            print(answer)
            print(translate_baidu(answer))
            generateSound(answer, id, model_id)
            PlaySound(r'.\output.wav', flags=1)
        elif model_id == 2:
            resp = api.send_message(get_input())
            if(resp == "quit()"):
                break
            answer = resp["message"].replace('\n','')
            print("ChatGPT:")
            print(answer)


            generateSound(answer, id, model_id)
            PlaySound(r'.\output.wav', flags=1)

