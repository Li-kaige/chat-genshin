print("加载torch...")
#所有库
import torch
import numpy
from tqdm import tqdm
from scipy.io.wavfile import write
import time
import subprocess
import datetime
#其他包内容
import commons2
import utils2
# from data_utils2 import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models2 import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence2
#定义的函数
def get_text(text, hps):
    text_norm = text_to_sequence2(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons2.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
def gettime():
    curr_time=datetime.datetime.now()
    return curr_time.strftime("%Y_%m_%d_%H_%M_%S_%f")
print("加载完成")
print("加载模型和配置")
hps_mt = utils2.get_hparams_from_file("./model/genshin/genshin.json")
#定义网络，加载模型
net_g_mt = SynthesizerTrn(
    len(symbols),
    hps_mt.data.filter_length // 2 + 1,
    hps_mt.train.segment_size // hps_mt.data.hop_length,
    n_speakers=hps_mt.data.n_speakers,
    **hps_mt.model).cuda()
_ = net_g_mt.eval()
_ = utils2.load_checkpoint("./model/genshin/genshin.pth", net_g_mt, None)
print("加载完成")
#清屏
i=subprocess.call("cls", shell=True)
#人物列表
npcList = ['派蒙', '凯亚', '安柏', '丽莎', '琴', '香菱', '枫原万叶',
           '迪卢克', '温迪', '可莉', '早柚', '托马', '芭芭拉', '优菈',
           '云堇', '钟离', '魈', '凝光', '雷电将军', '北斗',
           '甘雨', '七七', '刻晴', '神里绫华', '戴因斯雷布', '雷泽',
           '神里绫人', '罗莎莉亚', '阿贝多', '八重神子', '宵宫',
           '荒泷一斗', '九条裟罗', '夜兰', '珊瑚宫心海', '五郎',
           '散兵', '女士', '达达利亚', '莫娜', '班尼特', '申鹤',
           '行秋', '烟绯', '久岐忍', '辛焱', '砂糖', '胡桃', '重云',
           '菲谢尔', '诺艾尔', '迪奥娜', '鹿野院平藏']
girl_npcList = ['派蒙', '丽莎', '琴', '香菱', '可莉', '早柚',  '芭芭拉', '优菈', '云堇', '凝光', '雷电将军', '北斗',
                '甘雨', '七七', '刻晴', '神里绫华', '罗莎莉亚', '八重神子', '宵宫', '九条裟罗', '夜兰', '珊瑚宫心海',
                '女士', '莫娜', '申鹤', '烟绯', '久岐忍', '辛焱', '砂糖', '胡桃','菲谢尔', '诺艾尔', '迪奥娜']
boy_npcList=list(set(npcList)-set(girl_npcList))
new_npcList=girl_npcList+boy_npcList
#生成
print("【转换菜单】")
print("【人物列表】")
for i in range(len(new_npcList)):
    print(str(i)+":"+str(new_npcList[i]),end=" ")
print("")
index=1
while True:
    print("第【{}】轮".format(index))
    index=index+1
    speaker=input("请输入朗说话人的编号，输入-1退出，输入-2根据列表逐个生成，输入-3根据列表统一生成：")
    if speaker == "-1":
        break
    elif speaker=="-2" or speaker=="-3":
        wordlist=input("请输入列表的名称(xxx.txt)，前面是说话人编号/名称，后面是句子：")
        lines=open(wordlist,"r",encoding='utf8').readlines()
        speakers=[]
        contents=[]
        for line in lines:
            splits=line.split(" ")
            if splits[0].isdigit() == True:
                speakers.append(new_npcList[int(splits[0])])
            else:
                speakers.append(splits[0])
            contents.append(splits[1].replace("\n", "").replace('\u3000',''))
        audio_all = numpy.array([])
        for i in tqdm(range(len(contents))):
            phoneme = get_text(contents[i].replace("\n", "").replace('\u3000',''), hps_mt)
            audio = numpy.array([])
            with torch.no_grad():
                input_text = phoneme.cuda().unsqueeze(0)
                input_len = torch.LongTensor([phoneme.size(0)]).cuda()
                sid = torch.LongTensor([npcList.index(speakers[i])]).cuda()
                audio =net_g_mt.infer(input_text, input_len, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()
                audio_all=numpy.concatenate((audio_all,audio))
            if speaker == "-2":
                outputfilename="output/"+gettime()+"_"+speakers[i]+"_"+contents[i]+".wav"
                write(outputfilename, hps_mt.data.sampling_rate, audio)
        if speaker == "-3":
            outputfilename="output/"+gettime()+"_"+"长音频"+".wav"
            write(outputfilename, hps_mt.data.sampling_rate, audio_all)
        continue
    else:
        if speaker.isdigit() == True:
            speaker=new_npcList[int(speaker)]
        else:
            speaker=speaker
    sentence=input("请输入将要转换的语句：")
    print("开始合成，说话人=",speaker,"内容=",sentence)
    T1 = time.time()
    phoneme = get_text(sentence.replace("\n", "").replace('\u3000',''), hps_mt)
    audio = numpy.array([])
    with torch.no_grad():
        input_text = phoneme.cuda().unsqueeze(0)
        input_len = torch.LongTensor([phoneme.size(0)]).cuda()
        sid = torch.LongTensor([npcList.index(speaker)]).cuda()
        audio =net_g_mt.infer(input_text, input_len, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()
    outputfilename="output/"+gettime()+"_"+speaker+"_"+sentence+".wav"
    write(outputfilename, hps_mt.data.sampling_rate, audio)
    T2=time.time()
    print("合成成功，已保存在output目录中，耗时：",(T2-T1),"秒")
print("谢谢使用，再见！")



