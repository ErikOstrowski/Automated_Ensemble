import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

ERIK = 0.9


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='eriknet.0@png', type=str)
parser.add_argument("--domain", default='one_CT', type=str)
parser.add_argument("--threshold", default=0.5, type=float)

parser.add_argument("--predict_dir", default='/home/erik/CT_model', type=str)
parser.add_argument('--gt_dir', default='/srv/data/eostrowski/Dataset/brats3_2019/gt/', type=str)#'#GT_VOC_benign/', type=str)

parser.add_argument('--logfile', default='',type=str)
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--mode', default='npy', type=str) # png
parser.add_argument('--max_th', default=1.00, type=float)

args = parser.parse_args()

predict_folder = '/home/erik/CT_model/'
musks = './brats_cor_res34_th_3/' # deca_load_res50_2/'  # 
musks2 = './brats_cor_res50_th_3/' # #'./brats_load_res50_train_3/'#Grad-CAM_train/deca_swav_imgnet_3/'#deca_swav_out_50_3_imagenet/' #flood_05_12_th=70_3/' #flood_USS_deca_load_th=70_2/' #deca_load_out_70_3/' #flood_USS_deca_load_th=70_1/' #flood_USS_deca_load_th=50_1/'#deca_load_out_70_3/' #'flood_USS_deca_01_12_th=50/' <== best so far #  flood_USS_deca_30_11/' # deca_28_th=50/'#flood_deca/'   flood_USS_deca_01_12
#musks =  '/home/erik/PycharmProjects/Test/Grad-CAM-pytorch/

predict_folder = musks
gt_folder = args.gt_dir

args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

categories = ['benign', 'malignant']
num_cls = len(categories)

#musks = '/home/erik/PycharmProjects/Test/USS/predicts_cam/' #predicts_05_255_alt_001_cam/'
#musks2 = '/home/erik/PycharmProjects/Test/USS/orig_cam/'
for tu in range(9):
 tu = (tu+1)/10
 def compare(start,step,TP,P,T, name_list):
    eriki=0
    #for idx in range(start,len(name_list),step):

     
    for obj in os.listdir(gt_folder): # musks
        #name ='benign (1)'# name_list[idx] # '2007_000033'

        name = obj[:-4]
        #print(name)
        pred_name = 'image' + name[4:]
        gt_name = obj
        #full_musks2 = musks + pred_name + '.png'
        full_musks2 = musks + pred_name + '.npy'
        full_musks3 = musks2 + pred_name  +'.npy'
        #print(full_musks2)


        #if (os.path.isfile(full_musks)):
        #print(predict_folder + name + '.png')
        if os.path.isfile(predict_folder + pred_name + '.npy'):
                #if os.path.isfile(gt_folder + name + '.png'):
                #print(os.path.join(musks, name + '.npy'))
                #predict_dict = np.load(os.path.join(musks, name + '.npy'), allow_pickle=True).item()

                if True:#'hr_cam' in predict_dict.keys():
                    #eriki +=1
                    #print(eriki)
                    #cams = predict_dict['hr_cam']

                    full_musks = np.load(full_musks2)
                    
                    full_musks4 = np.load(full_musks3)

                    #full_musks = Image.open(full_musks2)
                    #full_musks4 = Image.open(full_musks3)
                    #newsize = (320, 320)
                    #full_musks = full_musks.resize(newsize)
                    

                    #print(full_musks.shape)
                    #full_musks = np.sum(full_musks, axis=0)
                    full_musks[full_musks > tu] = 255 ###################################tu] = 255
                    full_musks4[full_musks4 > ERIK] = 255
                    
                    full_musks[full_musks < 255] = 0
                    full_musks4[full_musks4 < 255] = 0
                    
                    
                    #################################################################################################################
                    # INVERTER
                    
                    #full_musks4[full_musks4 ==0] = 1
                    #full_musks4[full_musks4 ==255] = 0
                    #full_musks4[full_musks4 ==1] = 255
                    
                    #################################################################################################################
                    
                    
                    full_musks = full_musks + full_musks4
                    #print(full_musks.shape)
                    #img = Image.fromarray((full_musks).astype(np.uint8))
                    #img.show()
                    #break

                    #full_musks = np.sum(full_musks, axis=0)
                    #camsb = np.sum(cams, axis=0)

                    #cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
                    #zer = np.zeros(cams.shape)
                    #for a in range(cams.shape[0]):
                        #cams[a] = full_musks#cams[a]#*full_musks

                #print('OoO'*200)
                #keys = predict_dict['keys']
                #predict = keys[np.argmax(cams, axis=0)]
                predict = full_musks
                gt_file = os.path.join(gt_folder,gt_name[:-4] + '.npy')
                #print(gt_file)
                
                gt = np.load(gt_file)#Image.open(gt_file)
                ##newsize = (128, 128)
                ##gt = gt.resize(newsize)
                ##gt = np.array(gt)
                #predict[predict> 0.999999999] = 255
                #print(predict.shape)
                #predict= full_musks
                #zer = np.zeros(gt.shape)
                #if not(np.any(gt)):
                # 	#print('AYAYA'*100)
                # 	predict[predict>0]=0
                predict[predict>0]=1
                
                #predict[predict > 0] = 255
                #print(predict.shape)
                #print(gt.shape)
                gt[gt>0]= 1
                #predict = gt
                #img = Image.fromarray((predict).astype(np.uint8))
                #img.show()
                #break


                cal = gt<255
                #print(gt.shape)
                #print(predict.shape)
                #print(name)
                #print(np.amax(predict))
                #print(np.amax(cal))
                #print(gt.shape)
                #gt[gt > 0] = 255
                #img = Image.fromarray((gt).astype(np.uint8))
                #img.show()
                mask = (predict==gt) * cal
                #print (mask[mask==False])
                for i in range(num_cls):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    #print(P[i])
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
            
 def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    p_list = []
    for i in range(1):
        p = multiprocessing.Process(target=compare, args=(i,1,TP,P,T, name_list))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    DICE = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        #DICE.append(( TP[i].value + TP[i].value)/(T[i].value+P[i].value))

    loglist = {}
    for i in range(num_cls):
        # if i%2 != 1:
        #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
        # else:
        #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        loglist[categories[i]] = IoU[i] * 100
    
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    dic = np.mean(np.array(DICE)[1:])
    # print('\n======================================================')
    # print('%11s:%7.3f%%'%('mIoU',miou*100))
    # print('%11s:%7.3f'%('T/TP',t_tp))
    # print('%11s:%7.3f'%('P/TP',p_tp))
    # print('%11s:%7.3f'%('FP/ALL',fp_all))
    # print('%11s:%7.3f'%('FN/ALL',fn_all))
    # print('%11s:%7.3f'%('miou_foreground',miou_foreground))
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    #print(f'REAL DICE CLAY: {(2 * miou)/(miou +1) }')
    return loglist

 if __name__ == '__main__':
    #df = pd.read_csv(args.list, names=['filename'])
    name_list = ['0','1']#df['filename'].values

    if args.mode == 'png':
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 3)
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
    elif args.mode == 'rw':
        th_list = np.arange(0.05, args.max_th, 0.05).tolist()

        over_activation = 1.60
        under_activation = 0.60
        
        mIoU_list = []
        FP_list = []

        for th in th_list:
            args.threshold = th
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)

            mIoU, FP = loglist['mIoU'], loglist['fp_all']

            print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(th, mIoU, FP))

            FP_list.append(FP)
            mIoU_list.append(mIoU)
        
        best_index = np.argmax(mIoU_list)
        best_th = th_list[best_index]
        best_mIoU = mIoU_list[best_index]
        best_FP = FP_list[best_index]

        over_FP = best_FP * over_activation
        under_FP = best_FP * under_activation

        print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

        over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
        under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

        over_index = np.argmin(over_loss_list)
        over_th = th_list[over_index]
        over_mIoU = mIoU_list[over_index]
        over_FP = FP_list[over_index]

        under_index = np.argmin(under_loss_list)
        under_th = th_list[under_index]
        under_mIoU = mIoU_list[under_index]
        under_FP = FP_list[under_index]
        
        print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(best_th, best_mIoU, best_FP))
        print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
        print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
    else:
        if args.threshold is None:
            th_list = np.arange(0.05, 1.10, 0.05).tolist()
            
            best_th = 0
            best_mIoU = 0

            for th in th_list:
                args.threshold = th
                loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 2)
                print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))

                if loglist['mIoU'] > best_mIoU:
                    best_th = th
                    best_mIoU = loglist['mIoU']
            
            print('Best Th={:.2f}, mIoU={:.3f}%'.format(best_th, best_mIoU))
        else:
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 2)
            #print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
            print('{:.3f}'.format(loglist['mIoU']))

