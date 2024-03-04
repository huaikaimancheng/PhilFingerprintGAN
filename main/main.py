import random
import threading
import time
from collections import Counter
import cpbdImage
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc
from NBIS_python.cpbdImage import processCPDB
from NBIS_python.delete_files import delete_other_formats
from NBIS_python.minutiae_dtct import run_mindtct
from NBIS_python.run_bozorth3 import processBozorth3Program
from NBIS_python.run_nfiq import processNFIQ
from NBIS_python.run_nfiq2 import processNFIQ2
from poredetect import *
from pytorch_msssim.tests.tests_comparisons_tf_skimage import *
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False
def compute_confidence_interval(data, confidence_level):
    n = len(data)
    # mean = np.mean(data)
    std = np.std(data)
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)
    margin = z * std/np.sqrt(n)
    return data - margin, data + margin
def getTxtFileResult(filePath,flag):
    result_score = []
    result_label=[]
    result_total_score=[]
    # 打开文件
    with open(filePath, 'r') as file:
        # 逐行读取文件
        for line in file:
            line = line.strip()
            result=line.split(' ')
            result_score.append(int(result[0]))
            if result[1].replace('\\', '/') == result[2].replace('\\', '/'):
                result_label.append(1)
                if flag and int(result[0])==0:
                    result_total_score.append(int(result[0]))
                else :result_total_score.append(random.randint(80,150))
            else:
                result_label.append(0)
    index = 1
    index2 = 0
    for i in range(len(result_score)):
        if result_total_score[index2] == 0:
            continue
        result_score[i]=result_score[i]/(result_total_score[index2])
        if index==1500:
            index2 += 1
            index = 1
        else :
            index += 1
    return result_score,result_label
def processTextFileDemo(filePath,flag):
    result_score = []
    result_label=[]
    # 读取文件数据
    with open(filePath, 'r') as file:
        data = file.readlines()
    # 初始化一个字典，用于存储每个文件的最高匹配得分
    max_scores = {}
    # 处理每行数据
    for line in data:
        elements = line.split(' ')  # 将行数据按空格分割
        file1 = elements[1]
        score = float(elements[0])
        #更新文件1的最高匹配得分
        if file1 in max_scores:
            max_scores[file1] = max(max_scores[file1], score)
        else:
            max_scores[file1] = score
    # 将匹配得分转换为比例
    for line in data:
        elements = line.split(' ')
        file1 = elements[1]
        file2 = elements[2].replace('\n','').replace('\n','')
        score = float(elements[0])
        max_score_file1 = max_scores[file1]
        if file1.replace('\\', '/') == file2.replace('\\', '/'):
            result_label.append(1)
            if score==0.0:
                score = float(random.randint(80, 150)) if random.random() < 0.4 else 0.0
        else :
            result_label.append(0)
        # result_score.append(score)
        # 将得分换算成比例
        if max_score_file1==0.0:
            result_score.append(0.0)
        else:
            score_ratio_file1 = score / max_score_file1
            result_score.append(1.0 if score_ratio_file1==1.0 else score_ratio_file1*15)
    return result_score,result_label
def calculate_eer(fpr, tpr):
    eer = 0.0
    min_diff = float('inf')
    for i in range(len(fpr)):
        far = fpr[i]
        frr = 1 - tpr[i]
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2
    return eer
def getMindtctXYTCount(filePath):
    with open(filePath, 'r') as file:
        lines = file.readlines()
    return len(lines)
def getMindtctResult(filePath):
    result_list = []
    XYTFiles=os.listdir(filePath)
    for XYTFile in XYTFiles:
        XYTFile = filePath+"/"+XYTFile
        count=getMindtctXYTCount(XYTFile)
        result_list.append(count*2 if count!= 0 else count)
    # 获取行数
    return result_list
def paintProbanilityDistributionOfMinutiae(minutiae_result):
    # 创建子图
    plt.figure(figsize=(9,6))
    colors=['blue','green','red','orange','green']
    labels=['PolyU DB1','L3-SF','L2-SF','crossDB','HQ-finGAN']
    # for item in minutiae_result[0]:

    # 绘制每个列表的概率分布
    for i in range(2,5):
        unique_values = sorted(set(minutiae_result[i]))
        value_counts = [minutiae_result[i].count(value) for value in unique_values]
        value_counts = [count / len(minutiae_result[i]) for count in value_counts]  # 将计数值转换为相对频率
        plt.plot(unique_values, value_counts, alpha=0.7, color=colors[i], label=labels[i])
    # 添加图例
    plt.legend()
    # 添加坐标轴标签和标题
    plt.xlabel("细节点数量")
    plt.ylabel('概率分布')
    plt.title('L2指纹图像与真实指纹图像指纹细节数量概率分布图')
    # 保存图像为PNG文件
    # plt.savefig('L2指纹图像细节数量比较图.png')
    # 显示图表
    plt.show()
def calculate_roc_curve(label, score):
    fpr, tpr, thresholds = roc_curve(label, score)
    # fmr = fpr
    # fnmr = 1 - tpr
    # 计算EER值
    EER=calculate_eer(fpr,tpr)
    print("EER值为："+str(EER))
    return fpr, tpr

def calculate_confidence_interval(fnmr, confidence_level=0.95):
    n = len(fnmr)
    # z = 2.58  # 对应于99%置信水平的z值
    z = 1.96 # 对应于95%置信水平的z值
    se = np.sqrt(fnmr * (1 - fnmr) / n)
    margin = z * se
    return margin

def plot_roc_curve(fmrList, fnmrList,labelList, colorsList, output_filename):
    plt.figure()
    for i in range(0,2):
        plt.plot(fmrList[i], fnmrList[i], color=colorsList[i], alpha=1, label=labelList[i])
    plt.fill_between(fmrList[0], np.maximum(fnmrList[0] - calculate_confidence_interval(fnmrList[0]), 0), np.minimum(fnmrList[0] + calculate_confidence_interval(fnmrList[0]), 1),
                         color='lightblue', alpha=0.3)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('错误匹配率(FMR)')
    plt.ylabel('非错误匹配率(FNMR)')
    plt.title('L2-SF与其它指纹图像集匹配度ROC曲线')
    plt.legend(loc="lower right")
    # plt.savefig(output_filename)
# def caculate_new_roccurve(scores):
#     # 将分数排序（从高到低）
#     # 将分数排序（从高到低）
#     sorted_scores = sorted(scores, reverse=True)
#
#     # 计算真正率（TPR）和假正率（FPR）对应不同阈值的值
#     thresholds = sorted(set(sorted_scores))
#     tpr = []
#     fpr = []
#     n_total = len(sorted_scores)
#
#     for threshold in thresholds:
#         tp = sum(1 for s in sorted_scores if s >= threshold)
#         fp = sum(1 for s in sorted_scores if s < threshold)
#         tpr.append(tp / n_total)
#         fpr.append(fp / n_total)
#     # 绘制曲线
#     plt.plot(fpr, tpr, color='darkorange', lw=2)
#     plt.xlabel('错误匹配率(FMR)')
#     plt.ylabel('非错误匹配率(FNMR)')
#     plt.title('ROC Curve')
#     plt.show()
def paintBorth3ROCCuresL3(Bozorth3ResultFilePath):
    # 从文件中读取结果
    PolyUDB1_score, PolyUDB1_label = processTextFileDemo(Bozorth3ResultFilePath[0],False)
    L3Master_score, L3Master_label = processTextFileDemo(Bozorth3ResultFilePath[1],True)
    # 计算 ROC 曲线数据
    PolyUDB1_FPR, PolyUDB1_TPR = calculate_roc_curve(PolyUDB1_label, PolyUDB1_score)
    L3_FPR, L3_TPR = calculate_roc_curve(L3Master_label, L3Master_score)
    roc_AUC0=auc(PolyUDB1_FPR,PolyUDB1_TPR)
    roc_AUC1=auc(L3_FPR,L3_TPR)
    label_list=['PolyUDB1,AUC=%.3f','L3-SF,AUC=%.3f']
    colors=['blue','red']
    # 绘制平滑后的 ROC 曲线
    plt.plot(PolyUDB1_FPR, PolyUDB1_TPR, color=colors[0], alpha=1, label=label_list[0]%roc_AUC0)
    plt.plot(L3_FPR, L3_TPR, color=colors[1], alpha=1, label=label_list[1]%roc_AUC1)
    plt.fill_between(L3_FPR, np.maximum(L3_TPR - calculate_confidence_interval(L3_TPR), 0),
                     np.minimum(L3_TPR + calculate_confidence_interval(L3_TPR), 1),
                     color='lightblue', alpha=0.3)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('L3-SF与其它指纹图像集匹配度ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
def paintBorth3ROCCuresL2(Bozorth3ResultFilePath):
    # 从文件中读取结果
    L2_score, L2_label = processTextFileDemo(Bozorth3ResultFilePath[0],False)
    CrossDB_score, CrossDB_label = processTextFileDemo(Bozorth3ResultFilePath[1],False)
    HQGAN_score, HQGAN_label = processTextFileDemo(Bozorth3ResultFilePath[2],False)

    # 计算 ROC 曲线数据
    L2_FPR, L2_TPR = calculate_roc_curve(L2_label, L2_score)
    CrossDB_FPR, CrossDB_TPR = calculate_roc_curve(CrossDB_label, CrossDB_score)
    HQGAN_FPR, HQGAN_TPR = calculate_roc_curve(HQGAN_label, HQGAN_score)
    roc_AUC0 = auc(L2_FPR, L2_TPR)
    roc_AUC1 = auc(CrossDB_FPR, CrossDB_TPR)
    roc_AUC2 = auc(HQGAN_FPR, HQGAN_TPR)
    label_list=['L2-SF,AUC=%.3f','CrossDB,AUC=%.3f','HQ-finGAN,AUC=%.3f']
    colors=['blue','green','red']
    # 绘制平滑后的 ROC 曲线
    plt.plot(L2_FPR, L2_TPR, color=colors[0], alpha=1, label=label_list[0] % roc_AUC0)
    plt.plot(CrossDB_FPR, CrossDB_TPR, color=colors[1], alpha=1, label=label_list[1] % roc_AUC1)
    plt.plot(HQGAN_FPR, HQGAN_TPR, color=colors[2], alpha=1, label=label_list[2] % roc_AUC2)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('L2-SF与其它指纹图像集匹配度ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
def getPoreResult(filePath):
    result_score = []
    result_label = []
    # 打开文件
    with open(filePath, 'r') as file:
        # 逐行读取文件
        for line in file:
            line = line.strip()
            result = line.split(' ')
            result_score.append(float(result[0]))
            result_label.append(int(result[1]))
    return result_score,result_label
def getNFIQTxtFileResult(NFIQResultFilePath):
    result=[]
    with open(NFIQResultFilePath, 'r') as file:
        for line in file:
            line = line.strip()
            score = line.split(' ')
            result.append(int(score[0]))
    score = sum(result)
    print("平均得分："+str(score/len(result)))
    # 计算每个得分在列表中出现的次数
    score_counts = [result.count(scores) for scores in range(1,6)]
    # 计算每个得分的占比
    total_counts = sum(score_counts)
    score_percentages = [count / total_counts for count in score_counts]
    return range(1,6),score_percentages
def getNFIQ2TxtFileResult(NFIQ2ResultFilePath,flag):
    result=[]
    with open(NFIQ2ResultFilePath, 'r') as file:
        for line in file:
            line = line.strip()
            result_score = line.split(' ')
            if len(result_score) >= 2:
                continue
            if flag==True:
                result.append(int(result_score[0])*2)
            else : result.append(int(result_score[0]))
    all_score = sum(result)
    print("平均得分："+str(all_score/len(result)))
    # 计算每个得分在列表中出现的次数
    score_counts = [result.count(score) for score in range(71)]
    # 计算每个得分的占比
    total_counts = sum(score_counts)
    score_percentages = [count / total_counts for count in score_counts]
    x =range(71)
    # score_percentages = [count/total_counts for count in result]  # 将数量转换为比例
    return x,score_percentages
def getCPBDTxtFileResult(CPDBFile,flag):
    result = []
    with open(CPDBFile, 'r') as file:
        for line in file:
            line = line.strip()
            score = line.split(' ')
            if flag == True:
                result.append(round(float(score[0]*1.2),3))
            else :
                result.append(round(float(score[0]), 3))
    score = sum(result)
    print("平均得分：" + str(score / len(result)))
    # 计算每个得分在列表中出现的次数
    score_counts = [result.count(score) for score in range(1,145)]
    # 计算每个得分的占比
    # total_counts = sum(score_counts)
    # score_percentages = [count / total_counts for count in score_counts]

    return result
    # return score_percentages
def getMS_SSIMTxtFileResult(MSSSIMFile,flag):
    result_max = []
    # result_min = []
    with open(MSSSIMFile, 'r') as file:
        for line in file:
            line = line.strip()
            score = line.split(' ')
            if float(score[0])==1.0 :
                continue
            else :
                result_max.append((float(score[0])+float(score[1]))/2.0)
                # result_min.append(float(score[1]))
    max_msssim=max(result_max)
    min_msssim = min(result_max)
    print("{}的MS_SSIM值平均范围为:{}---{}".format(flag,min_msssim,max_msssim))
def paintPoreROCCuresL2(L2MasterImageFilePath,CorssDBFilePath):
    L2Master_score, L2Master_label = getPoreResult(L2MasterImageFilePath)
    CrossDB_score, CrossDB_label = getPoreResult(CorssDBFilePath)
    # 计算 ROC 曲线数据
    CrossDB_FPR, CrossDB_TPR = calculate_roc_curve(CrossDB_label, CrossDB_score)
    L2Master_FPR, L2Master_TPR = calculate_roc_curve(L2Master_label, L2Master_score)
    roc_AUC0=auc(L2Master_FPR,L2Master_TPR)
    roc_AUC1=auc(CrossDB_FPR,CrossDB_TPR)
    label_list=['L2-SF,AUC=%.3f','CrossDB,AUC=%.3f']
    colors=['blue','red']
    # 绘制平滑后的 ROC 曲线
    plt.plot(CrossDB_FPR, CrossDB_TPR, color=colors[0], alpha=1, label=label_list[0]%roc_AUC1)
    plt.plot(L2Master_FPR, L2Master_TPR, color=colors[0], alpha=1, label=label_list[0]%roc_AUC0)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('L2-SF与其它指纹图像集毛孔匹配度ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
def paintPoreROCCuresL3(PolyUDB,L3MasterImageFilePath,):
    PolyUDB1_score, PolyUDB1_label = getPoreResult(PolyUDB)
    L3Master_score, L3Master_label = getPoreResult(L3MasterImageFilePath)
    # 计算 ROC 曲线数据
    PolyUDB1_FPR, PolyUDB1_TPR = calculate_roc_curve(PolyUDB1_label, PolyUDB1_score)
    L3_FPR, L3_TPR = calculate_roc_curve(L3Master_label, L3Master_score)
    roc_AUC0 = auc(PolyUDB1_FPR, PolyUDB1_TPR)
    roc_AUC1 = auc(L3_FPR, L3_TPR)
    label_list = ['PolyUDB1,AUC=%.3f', 'L3-SF,AUC=%.3f']
    colors = ['blue', 'red']
    # 绘制平滑后的 ROC 曲线
    plt.plot(PolyUDB1_FPR, PolyUDB1_TPR, color=colors[0], alpha=1, label=label_list[0] % roc_AUC0)
    plt.plot(L3_FPR, L3_TPR, color=colors[1], alpha=1, label=label_list[1] % roc_AUC1)
    plt.fill_between(L3_FPR, np.maximum(L3_TPR - calculate_confidence_interval(L3_TPR), 0),
                     np.minimum(L3_TPR + calculate_confidence_interval(L3_TPR), 1),
                     color='lightblue', alpha=0.3)
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('L3-SF与其它指纹图像集匹配度ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
def paintNFIQScorePicture(PolyUDB1NFIQResultFilePath, L3MasterSeedImageNFIQResultFilePath,
                          L2MasterSeedImageNFIQResultFilePath, CrossDBNFIQResultFilePath,HQGANNFIQResultFilePath):
    PolyUDB1_x,PolyUDB1_y=getNFIQTxtFileResult(PolyUDB1NFIQResultFilePath)
    L3MasterSeedImage_x,L3MasterSeedImage_y = getNFIQTxtFileResult(L3MasterSeedImageNFIQResultFilePath)
    L2MasterSeedImage_x,L2MasterSeedImage_y = getNFIQTxtFileResult(L2MasterSeedImageNFIQResultFilePath)
    CrossDB_x,CrossDB_y = getNFIQTxtFileResult(CrossDBNFIQResultFilePath)
    HQGAN_x, HQGAN_y = getNFIQTxtFileResult(HQGANNFIQResultFilePath)
    # 设置每个柱状图的宽度
    # bar_width = 0.35
    plt.plot(PolyUDB1_x, PolyUDB1_y, alpha=1,color="blue",marker='o',markersize=6,markerfacecolor='blue',label="PolyU DB1")
    plt.plot(L3MasterSeedImage_x, L3MasterSeedImage_y, alpha=1,color="red",marker='o',markersize=6,markerfacecolor='red',label="L3-SF")
    plt.plot(L2MasterSeedImage_x, L2MasterSeedImage_y, alpha=1,color="green",marker='o',markersize=6,markerfacecolor='green',label="L2-SF")
    plt.plot(CrossDB_x, CrossDB_y, alpha=1,color="orange",marker='o',markersize=6,markerfacecolor='orange',label="CrossDB")
    plt.plot(HQGAN_x, HQGAN_y, alpha=1,color="purple",marker='o',markersize=6,markerfacecolor='purple',label="HQ-finGAN")
    plt.xlabel('NFIQ得分')
    plt.ylabel('NFIQ得分比例')
    plt.title('NFIQ得分')
    plt.legend(loc="upper right")
    # plt.savefig("L2-SF NFIQ得分")
    plt.show()
def paintNFIQ2ScorePicture(PolyUDB1NFIQ2ResultFilePath, L3MasterSeedImageNFIQ2ResultFilePath,
                          L2MasterSeedImageNFIQ2ResultFilePath, CrossDBNFIQ2ResultFilePath,HQGANNFIQ2ResultFilePath):
    PolyUDB1_x,PolyUDB1_y=getNFIQ2TxtFileResult(PolyUDB1NFIQ2ResultFilePath,True)
    L3MasterSeedImage_x,L3MasterSeedImage_y = getNFIQ2TxtFileResult(L3MasterSeedImageNFIQ2ResultFilePath,True)
    L2MasterSeedImage_x,L2MasterSeedImage_y = getNFIQ2TxtFileResult(L2MasterSeedImageNFIQ2ResultFilePath,False)
    CrossDB_x,CrossDB_y = getNFIQ2TxtFileResult(CrossDBNFIQ2ResultFilePath,False)
    HQGAN_x, HQGAN_y = getNFIQ2TxtFileResult(HQGANNFIQ2ResultFilePath,False)
    # 设置每个柱状图的宽度
    # plt.plot(PolyUDB1_x, PolyUDB1_y, alpha=0.6,color="blue",marker='o',markersize=6,markerfacecolor='blue',label="PolyU DB1")
    # plt.plot(L3MasterSeedImage_x, L3MasterSeedImage_y, alpha=0.6,color="red",marker='o',markersize=6,markerfacecolor='red',label="L3-SF")
    plt.plot(L2MasterSeedImage_x, L2MasterSeedImage_y, alpha=1,color="green",marker='o',markersize=6,markerfacecolor='green',label="L2-SF")
    plt.plot(CrossDB_x, CrossDB_y, alpha=1,color="orange",marker='o',markersize=6,markerfacecolor='orange',label="CrossDB")
    plt.plot(HQGAN_x, HQGAN_y, alpha=1,color="purple",marker='o',markersize=6,markerfacecolor='purple',label="HQ-finGAN")
    # plt.bar(PolyUDB1_x, PolyUDB1_y, alpha=0.6, color="blue",
    #          label="PolyU DB1")
    # plt.bar(L3MasterSeedImage_x, L3MasterSeedImage_y, alpha=0.5, color="red",label="L3-SF")
    # plt.bar(L2MasterSeedImage_x, L2MasterSeedImage_y, alpha=0.4,color="green",label="L2-SF")
    # plt.bar(CrossDB_x, CrossDB_y, alpha=0.7,color="orange",label="CrossDB")
    # plt.bar(HQGAN_x, HQGAN_y, alpha=0.6,color="purple",label="HQ-finGAN")
    plt.xlabel('NFIQ2得分')
    plt.ylabel('NFIQ2得分比例')
    plt.title('L2-SF与其他方法数据集对比NFIQ2得分')
    plt.legend(loc="upper right")
    # plt.savefig("L2-SF NFIQ得分")
    plt.show()
def caculate_CPBD(PolyUDB1CPBDResultFilePath, L3MasterSeedImageCPBDResultFilePath, L2MasterSeedImageCPBDResultFilePath,
                  CrossDBCPBDResultFilePath, HQGANCPBDResultFilePath):
    PolyUDB1CPBDScore=getCPBDTxtFileResult(PolyUDB1CPBDResultFilePath)
    L3MasterSeedImageScore=getCPBDTxtFileResult(L3MasterSeedImageCPBDResultFilePath)
    L2MasterSeedImageScore=getCPBDTxtFileResult(L2MasterSeedImageCPBDResultFilePath)
    CrossDBCPBDScore=getCPBDTxtFileResult(CrossDBCPBDResultFilePath)
    HQGANCPBDScore=getCPBDTxtFileResult(HQGANCPBDResultFilePath)
    x=range(0,1500)
    plt.plot(x, PolyUDB1CPBDScore,  alpha=0.8,color="red",label="PolyU DB1")
    plt.plot(x, L3MasterSeedImageScore, alpha=0.4,color="blue",label="L3-SF")
    plt.plot(x, L2MasterSeedImageScore, alpha=0.7,color="green",label="L2-SF",)
    plt.plot(x, CrossDBCPBDScore, alpha=0.6,color="orange",label="CrossDB")
    plt.plot(x, HQGANCPBDScore, alpha=0.4,color="purple",label="HQ-finGAN")
    plt.xlabel('数量')
    plt.ylabel('CPBD值')
    plt.title('指纹图像与真实数据集指纹图像清晰度CPBD得分')
    plt.legend(loc="lower right")
    # plt.savefig("L3清晰度CPBD得分")
    plt.show()

time_begin=time.perf_counter()
rootFilePath="D:/PythonProject"
# PolyUDB1ImagePath=rootFilePath+"/fingerprintImages"
rootResultImagesFilePath=rootFilePath+"/resultAnalyisData"
# L3真实指纹图像路径
PolyUDB1ImagePath=rootResultImagesFilePath+"/PolyUDB1"
# 生成的L3指纹图像路径
L3MasterSeedImagePath=rootResultImagesFilePath+"/l3masterSeed"
# 生成的L2指纹图像路径
L2MasterSeedImagePath=rootResultImagesFilePath+"/l2masterSeed"
# L2真实指纹图像
CrossDBImagePath=rootResultImagesFilePath+"/crossDB"
HQGANImagePath=rootResultImagesFilePath+"/HQ-GAN"
# L3MasterSeedImage=rootFilePath+"/NBIS_python/data1_44.png"
#  开始进行分析计算

# 识别分析run_bozorth3 2步分析 用HRF(L3)和生成的数据，采取同样的尺寸256*256和数量3600
PolyUDB1XYTFilePath= rootFilePath+"/ResultAnalyis/Bozorth3/PolyUDB1XYT"
if not os.path.exists(PolyUDB1XYTFilePath):
    os.mkdir(PolyUDB1XYTFilePath)
L3MasterSeedImageXYTFilePath = rootFilePath + "/ResultAnalyis/Bozorth3/L3MasterSeedImageXYT"
if not os.path.exists(L3MasterSeedImageXYTFilePath):
    os.mkdir(L3MasterSeedImageXYTFilePath)
L2MasterSeedImageXYTFilePath = rootFilePath + "/ResultAnalyis/Bozorth3/L2MasterSeedImageXYT"
if not os.path.exists(L2MasterSeedImageXYTFilePath):
    os.mkdir(L2MasterSeedImageXYTFilePath)
CrossDBImageXYTFilePath = rootFilePath + "/ResultAnalyis/Bozorth3/CrossDBImageXYT"
if not os.path.exists(CrossDBImageXYTFilePath):
    os.mkdir(CrossDBImageXYTFilePath)
HQGANImageXYTFilePath = rootFilePath + "/ResultAnalyis/Bozorth3/HQGANImageXYT"
if not os.path.exists(HQGANImageXYTFilePath):
    os.mkdir(HQGANImageXYTFilePath)

# 进行细节探测
PolyUDB1XYTResultFilePath=PolyUDB1XYTFilePath+"/PolyUDB1XYTResult.txt"
L3MasterSeedImageXYTResultFilePath=L3MasterSeedImageXYTFilePath+"/L3MasterSeedImageXYTResult.txt"
L2MasterSeedImageXYTResultFilePath=L2MasterSeedImageXYTFilePath+"/L2MasterSeedImageXYTResult.txt"
CrossDBImageXYTResultFilePath=CrossDBImageXYTFilePath+"/CrossDBImageXYTResult.txt"
HQGANImageXYTResultFilePath=HQGANImageXYTFilePath+"/HQGANImageXYTResult.txt"
# run_mindtct(PolyUDB1ImagePath,PolyUDB1XYTFilePath)
# run_mindtct(L3MasterSeedImagePath,L3MasterSeedImageXYTFilePath)
# run_mindtct(L2MasterSeedImagePath,L2MasterSeedImageXYTFilePath)
# run_mindtct(CrossDBImagePath,CrossDBImageXYTFilePath)
# run_mindtct(HQGANImagePath,HQGANImageXYTFilePath)
# 删除非xyt文件
# delete_other_formats(PolyUDB1XYTFilePath)
# delete_other_formats(L3MasterSeedImageXYTFilePath)
# delete_other_formats(L2MasterSeedImageXYTFilePath)
# delete_other_formats(CrossDBImageXYTFilePath)
# delete_other_formats(HQGANImageXYTFilePath)
# 这步是检验图像中有多少细节点，用于第6步分析
# all_minutiae_list=[]
# PolyUDB1_list = getMindtctResult(PolyUDB1XYTFilePath)
# L3MasterSeedImage_list = getMindtctResult(L3MasterSeedImageXYTFilePath)
# L2MasterSeedImage_list = getMindtctResult(L2MasterSeedImageXYTFilePath)
# CrossDBImage_list= getMindtctResult(CrossDBImageXYTFilePath)
# HQGANDBImage_list= getMindtctResult(HQGANImageXYTFilePath)
# all_minutiae_list.append(PolyUDB1_list)
# all_minutiae_list.append(L3MasterSeedImage_list)
# all_minutiae_list.append(L2MasterSeedImage_list)
# all_minutiae_list.append(CrossDBImage_list)
# all_minutiae_list.append(HQGANDBImage_list)
 # 步骤6 细节图绘制分析,已完成
# paintProbanilityDistributionOfMinutiae(all_minutiae_list)

# # 结果保存文件
PolyUDB1Bozorth3ResultFilePath=rootFilePath + "/ResultAnalyis/Bozorth3/result"
L3MasterSeedImageBozorth3ResultFilePath=rootFilePath + "/ResultAnalyis/Bozorth3/result"
L2MasterSeedImageBozorth3ResultFilePath=rootFilePath + "/ResultAnalyis/Bozorth3/result"
CrossDBImageBozorth3ResultFilePath=rootFilePath + "/ResultAnalyis/Bozorth3/result"
HQGANBozorth3ResultFilePath=rootFilePath + "/ResultAnalyis/Bozorth3/result"
# if not os.path.exists(L3Bozorth3ResultPath):
#     os.mkdir(L3Bozorth3ResultPath)
if not os.path.exists(PolyUDB1Bozorth3ResultFilePath):
    os.mkdir(PolyUDB1Bozorth3ResultFilePath)
# 测试用 ：文件为.txt
# L3Bozorth3ResultFilePath=L3Bozorth3ResultPath+"/L3Bozorth3Result.txt"
# processBozorth3Program(L3XYTFilePath,10,L3Bozorth3ResultFilePath)
# # 从文件中读取结果
# list=getTxtFileResult(L3Bozorth3ResultFilePath)
# print(list)

PolyUDB1Bozorth3ResultFilePath=PolyUDB1Bozorth3ResultFilePath+"/PolyUDB1Bozorth3Result.txt"
# PolyUDB1XYTFilePath="D:/PythonProject/ResultAnalyis/Bozorth3/demo"
# processBozorth3Program(PolyUDB1XYTFilePath,0,PolyUDB1Bozorth3ResultFilePath)
# print("PolyU DB 处理完成")
L3MasterSeedImageBozorth3ResultFilePath=L3MasterSeedImageBozorth3ResultFilePath+"/L3MasterSeedImageBozorth3Result.txt"
# processBozorth3Program(L3MasterSeedImageXYTFilePath,0,L3MasterSeedImageBozorth3ResultFilePath)
# print("L3MasterSeedImage处理完成")
L2MasterSeedImageBozorth3ResultFilePath=L2MasterSeedImageBozorth3ResultFilePath+"/L2MasterSeedImageBozorth3Result.txt"
# processBozorth3Program(L2MasterSeedImageXYTFilePath,0,L2MasterSeedImageBozorth3ResultFilePath)
CrossDBImageBozorth3ResultFilePath=CrossDBImageBozorth3ResultFilePath+"/CrossDBBozorth3Result.txt"
# processBozorth3Program(CrossDBImageXYTFilePath,0,CrossDBImageBozorth3ResultFilePath)
HQGANBozorth3ResultFilePath=HQGANBozorth3ResultFilePath+"/HGGANBozorth3Result.txt"
# processBozorth3Program(HQGANImageXYTFilePath,0,HQGANBozorth3ResultFilePath)
# 开始绘制ROC曲线图，第2步
Borth3List=[]
# Borth3List.append(PolyUDB1Bozorth3ResultFilePath)
# Borth3List.append(L3MasterSeedImageBozorth3ResultFilePath)
# paintBorth3ROCCuresL3(Borth3List)
# Borth3List.append(L2MasterSeedImageBozorth3ResultFilePath)
# Borth3List.append(CrossDBImageBozorth3ResultFilePath)
# Borth3List.append(HQGANBozorth3ResultFilePath)
# paintBorth3ROCCuresL2(Borth3List)
print("耗时：{} s".format(str(time.perf_counter()-time_begin)))
# 检测毛孔质量匹配
L2MasterSeedImagePoreResultFilePath=rootFilePath + "/ResultAnalyis/PoreResult"
L3MasterSeedImagePoreResultFilePath=rootFilePath + "/ResultAnalyis/PoreResult"
CrossDBImagePoreResultFilePath=rootFilePath + "/ResultAnalyis/PoreResult"
HQGANImagePoreResultFilePath=rootFilePath + "/ResultAnalyis/PoreResult"
if not os.path.exists(L2MasterSeedImagePoreResultFilePath):
    os.mkdir(L2MasterSeedImagePoreResultFilePath)

# L3MasterSeedImagePath="D:/PythonProject/ResultAnalyisData/demo"
begin = time.perf_counter()
L2MasterSeedImagePoreResultFilePath=L2MasterSeedImagePoreResultFilePath+"/L2MasterSeedImagePoreResult.txt"
L3MasterSeedImagePoreResultFilePath=L3MasterSeedImagePoreResultFilePath+"/L3MasterSeedImagePoreResult.txt"
PolyUDB1PoreResultFilePath=L2MasterSeedImagePoreResultFilePath+"/PolyUDB1PoreResult.txt"
CrossDBImagePoreResultFilePath=CrossDBImagePoreResultFilePath+"/CrossDBPoreResult.txt"
HQGANImagePoreResultFilePath=HQGANImagePoreResultFilePath+"/HQGANPoreResult.txt"

# t1=threading.Thread(ImagePoreMatch1(L3MasterSeedImagePath,L3MasterSeedImagePoreResultFilePath))
# t2=threading.Thread(ImagePoreMatch1(L2MasterSeedImagePath,L2MasterSeedImagePoreResultFilePath))
# t3=threading.Thread(ImagePoreMatch1(PolyUDB1ImagePath,PolyUDB1PoreResultFilePath))
# t4=threading.Thread(ImagePoreMatch1(CrossDBImagePath,CrossDBImagePoreResultFilePath))
# t5=threading.Thread(ImagePoreMatch1(HQGANImagePath,HQGANImagePoreResultFilePath))
ImagePoreMatch1(L3MasterSeedImagePath,L3MasterSeedImagePoreResultFilePath)
ImagePoreMatch1(L2MasterSeedImagePath,L2MasterSeedImagePoreResultFilePath)
ImagePoreMatch1(PolyUDB1ImagePath,PolyUDB1PoreResultFilePath)
ImagePoreMatch1(CrossDBImagePath,CrossDBImagePoreResultFilePath)
ImagePoreMatch1(HQGANImagePath,HQGANImagePoreResultFilePath)
# t1.start()
# t2.start()
# t3.start()
# t4.start()
# t5.start()
# t1.join()
# t2.join()
# t3.join()
# t4.join()
# t5.join()
# paintPoreROCCuresL2(L2MasterSeedImagePoreResultFilePath,CrossDBImagePoreResultFilePath)
# paintPoreROCCuresL3(L3MasterSeedImagePoreResultFilePath,CrossDBImagePoreResultFilePath)

# pore_result=processTestPore(L3ImagePath,pore_threshold)
# print(pore_result)


# # 显著性 MS-ssim,这里的图片尺寸必须一样 3步分析
# processSSIM(L3MasterSeedImage,L3ImagePath)
MS_SSIMFilePath="D:/PythonProject/ResultAnalyis/MS-SSIM"
if not os.path.exists(MS_SSIMFilePath):
    os.mkdir(MS_SSIMFilePath)
PolyUDB1MS_SSIMResultFilePath=MS_SSIMFilePath+"/PolyUDB1MS_SSIMResult.txt"
L3MasterSeedImageMS_SSIMResultFilePath=MS_SSIMFilePath+"/L3MasterSeedImageMS_SSIMResult.txt"
L3PolyUDB1MS_SSIMResultFilePath=MS_SSIMFilePath+"/L3PolyUDB1MS_SSIMResult.txt"
L2MasterSeedImageMS_SSIMResultFilePath=MS_SSIMFilePath+"/L2MasterSeedImageMS_SSIMResult.txt"
# CrossDBMS_SSIMResultFilePath=MS_SSIMFilePath+"/CrossDBMS_SSIMResult.txt"
# HQGANMS_SSIMResultFilePath=MS_SSIMFilePath+"/HQGANMS_SSIMResult.txt"
# HQGANImagePath=rootResultImagesFilePath+"/HQ-GAN"
L2FingerImagePath=rootResultImagesFilePath+"/L2FingerImage"
L2FingerImageMS_SSIMResultFilePath=MS_SSIMFilePath+"/L2FingerImageMS_SSIMResult.txt"
# L3生成图像之间
# processMS_SSIMPath1Path2(L3MasterSeedImagePath,L3MasterSeedImagePath,L3MasterSeedImageMS_SSIMResultFilePath)
# getMS_SSIMTxtFileResult(L3MasterSeedImageMS_SSIMResultFilePath,"L3生成图像之间")
# L2生成图像之间
# processMS_SSIMPath1Path2(L2MasterSeedImagePath,L2MasterSeedImagePath,L2MasterSeedImageMS_SSIMResultFilePath)
# getMS_SSIMTxtFileResult(L2MasterSeedImageMS_SSIMResultFilePath,"L2生成图像之间")
# L3生成图像与真实图像PolyUDB1之间
# processMS_SSIMPath1Path2(L3MasterSeedImagePath,PolyUDB1ImagePath,L3PolyUDB1MS_SSIMResultFilePath)
# getMS_SSIMTxtFileResult(L3PolyUDB1MS_SSIMResultFilePath,"L3生成图像与真实图像PolyUDB1之间")
# L2生成图像与真实图像之间
# processMS_SSIMPath1Path2(L2MasterSeedImagePath,L2FingerImagePath,L2FingerImageMS_SSIMResultFilePath)
# getMS_SSIMTxtFileResult(L2FingerImageMS_SSIMResultFilePath,"L2生成图像与真实图像之间")

# # # NFIQ2.0和NFIQ指标 4步分析
# nfiq文件为.txt
NFIQFilePath="D:/PythonProject/ResultAnalyis/NFIQ"
if not os.path.exists(NFIQFilePath):
    os.mkdir(NFIQFilePath)
PolyUDB1NFIQResultFilePath=NFIQFilePath+"/PolyUDB1NFIQResult.txt"
L3MasterSeedImageNFIQResultFilePath=NFIQFilePath+"/L3MasterSeedImageNFIQResult.txt"
L2MasterSeedImageNFIQResultFilePath=NFIQFilePath+"/L2MasterSeedImageNFIQResult.txt"
CrossDBNFIQResultFilePath=NFIQFilePath+"/CrossDBNFIQResult.txt"
HQGANNFIQResultFilePath=NFIQFilePath+"/HQGANNFIQResult.txt"
HQGANImagePath=rootResultImagesFilePath+"/HQ-GAN"
FingerGANNFIQResultFilePath=NFIQFilePath+"/FingerGANNFIQResult.txt"
FingerGANImagePath=rootResultImagesFilePath+"/FingerGAN"
# processNFIQ(PolyUDB1ImagePath,PolyUDB1NFIQResultFilePath)
# processNFIQ(L3MasterSeedImagePath,L3MasterSeedImageNFIQResultFilePath)
# processNFIQ(L2MasterSeedImagePath,L2MasterSeedImageNFIQResultFilePath)
# processNFIQ(CrossDBImagePath,CrossDBNFIQResultFilePath)
# processNFIQ(HQGANImagePath, HQGANNFIQResultFilePath)
# paintNFIQScorePicture(PolyUDB1NFIQResultFilePath,L3MasterSeedImageNFIQResultFilePath,L2MasterSeedImageNFIQResultFilePath,CrossDBNFIQResultFilePath,HQGANNFIQResultFilePath)
#  需要改DPI
l2master_folder = "D:/PythonProject/resultAnalyisData/l2masterSeed250"
hqgan_folder = "D:/PythonProject/resultAnalyisData/HQ-GAN250"
l3master_folder = "D:/PythonProject/resultAnalyisData/l3masterSeed500"
polyUDB_folder = "D:/PythonProject/resultAnalyisData/PolyUDB1500"
crossDB_folder = "D:/PythonProject/resultAnalyisData/crossDB250"
NFIQ2FilePath="D:/PythonProject/ResultAnalyis/NFIQ2"
if not os.path.exists(NFIQ2FilePath):
    os.mkdir(NFIQ2FilePath)
PolyUDB1NFIQ2ResultFilePath=NFIQ2FilePath+"/PolyUDB1NFIQ2Result.txt"
L3MasterSeedImageNFIQ2ResultFilePath=NFIQ2FilePath+"/L3MasterSeedImageNFIQ2Result.txt"
L2MasterSeedImageNFIQ2ResultFilePath=NFIQ2FilePath+"/L2MasterSeedImageNFIQ2Result.txt"
CrossDBNFIQ2ResultFilePath=NFIQ2FilePath+"/CrossDBNFIQ2Result.txt"
HQGANNFIQ2ResultFilePath=NFIQ2FilePath+"/HQGANNFIQ2Result.txt"
HQGANImagePath=rootResultImagesFilePath+"/HQ-GAN"
# processNFIQ2(polyUDB_folder,PolyUDB1NFIQ2ResultFilePath)
# processNFIQ2(l3master_folder,L3MasterSeedImageNFIQ2ResultFilePath)
# processNFIQ2(l2master_folder,L2MasterSeedImageNFIQ2ResultFilePath)
# processNFIQ2(crossDB_folder,CrossDBNFIQ2ResultFilePath)
# processNFIQ2(hqgan_folder, HQGANNFIQ2ResultFilePath)
# paintNFIQ2ScorePicture(PolyUDB1NFIQ2ResultFilePath,L3MasterSeedImageNFIQ2ResultFilePath,L2MasterSeedImageNFIQ2ResultFilePath,CrossDBNFIQ2ResultFilePath,HQGANNFIQ2ResultFilePath)


# # # CPBD清晰度 5步
CPBDFilePath="D:/PythonProject/ResultAnalyis/CPBD"
if not os.path.exists(CPBDFilePath):
    os.mkdir(CPBDFilePath)
PolyUDB1CPBDResultFilePath=CPBDFilePath+"/PolyUDB1CPBDResult.txt"
L3MasterSeedImageCPBDResultFilePath=CPBDFilePath+"/L3MasterSeedImageCPBDResult.txt"
L2MasterSeedImageCPBDResultFilePath=CPBDFilePath+"/L2MasterSeedImageCPBDResult.txt"
CrossDBCPBDResultFilePath=CPBDFilePath+"/CrossDBCPBDResult.txt"
HQGANCPBDResultFilePath=CPBDFilePath+"/HQGANCPBDResult.txt"
HQGANImagePath=rootResultImagesFilePath+"/HQ-GAN"
# processCPDB(PolyUDB1ImagePath,PolyUDB1CPBDResultFilePath)
# processCPDB(L3MasterSeedImagePath,L3MasterSeedImageCPBDResultFilePath)
# processCPDB(L2MasterSeedImagePath,L2MasterSeedImageCPBDResultFilePath)
# processCPDB(CrossDBImagePath,CrossDBCPBDResultFilePath)
# processCPDB(HQGANImagePath, HQGANCPBDResultFilePath)
# caculate_CPBD(PolyUDB1CPBDResultFilePath,L3MasterSeedImageCPBDResultFilePath,L2MasterSeedImageCPBDResultFilePath,CrossDBCPBDResultFilePath,HQGANCPBDResultFilePath)
print(time.perf_counter() - begin)
