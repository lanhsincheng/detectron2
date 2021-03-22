from detectron2.utils.visualizer import ColorMode
import cv2
import random
from detectron2.utils.visualizer import Visualizer
from projects.mammography_project.mammo_dataset import *
import operator
import xlsxwriter
import csv

wb_name = 'mammo0824_model_0059999'
def mammo_integrate(test_dirname, predictor, dataset_metadata, test_data_csv_path, output_dir):

    dataset_dicts = get_mammo_dicts(test_dirname,test_data_csv_path)
    answer_sheet_list = []
    for_ensemble_confidence_list = []
    score_class_list = []
    big_list = []
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        num_instances = len(outputs['instances'])
        scores_class_dict = {}
        for_ensemble_dict = {}
        per_class_dict = {}
        for s in range(num_instances):
            scores = outputs['instances']._fields['scores'].T[s].item()
            pred_classes_num = outputs['instances']._fields['pred_classes'].T[s].item()
            per_class_dict.update({pred_classes_num : scores})
            # if pred_classes_num == 0 or pred_classes_num == 2 or pred_classes_num == 4 or pred_classes_num == 6:
            #     pred_classes = 'benign'
            # elif pred_classes_num == 1 or pred_classes_num == 3 or pred_classes_num == 5 or pred_classes_num == 7:
            #     pred_classes = 'malignant'
            if pred_classes_num == 0 :
                pred_classes = 'benign'
            elif pred_classes_num == 1 :
                pred_classes = 'malignant'
            # if pred_classes_num == 1:
            #     pred_classes = 'benign'
            # elif pred_classes_num == 0:
            #     pred_classes = 'malignant'
            scores_class_dict.update( {scores : pred_classes} )
            if (pred_classes not in for_ensemble_dict) or (pred_classes in for_ensemble_dict and for_ensemble_dict[pred_classes] < scores):
                for_ensemble_dict.update( {pred_classes : scores} )
            if ('benign' not in for_ensemble_dict):
                for_ensemble_dict.update({'benign': 0})
            if ('malignant' not in for_ensemble_dict):
                for_ensemble_dict.update({'malignant': 0})
        per_class_dict = sorted(per_class_dict.items(), key= lambda per_class_dict: per_class_dict[0])
        big_list.append(per_class_dict)

        if ( not bool(scores_class_dict)==True ):
            print(scores_class_dict)
            scores_class_dict.update({0: 'benign', 0: 'malignant'})
            print(scores_class_dict)
        score_class_list.append(scores_class_dict)
        final_for_ensemble = sorted(for_ensemble_dict.items(), key=lambda for_ensemble_dict: for_ensemble_dict[0])
        final_class = max(scores_class_dict.items(), key=lambda scores_class_dict: scores_class_dict[0])[1]
        answer_sheet_list.append(final_class)
        for i in range(len(final_for_ensemble)):
            for_ensemble_confidence_list.append(final_for_ensemble[i][1])

    # load golden answer and evaluate accuracy
    golden = []
    golden_sheet = r'D:\Mammograph\golden/balance_golden_v1_852.csv'
    with open(golden_sheet, newline='') as csvFile:
        T = 0
        F = 0
        mm = 0
        bb = 0
        bm = 0
        mb = 0
        rows = csv.reader(csvFile)
        for row in rows:
            golden.append(row[0])
    for i, j in zip(golden, answer_sheet_list):
        if i == j:
            if(i=='benign'):
                bb += 1
            else:
                mm += 1
            T += 1
        else:
            if (i == 'benign'):
                bm += 1
            else:
                mb += 1
            F += 1
    accuracy = T / (T+F)
    accuracy_malignant =  (mm + bb)/(mm + bb + bm + mb)
    accuracy_benign = (bb + mm)/(bb + mm + mb + bm)
    print('T: ', T, ' F: ', F, ' accuracy: ',accuracy, 'malignant to benign : ', mb, 'benign to malignant : ', bm)

    # write class and confidence per bounding box to the xlsfile(4 classes)
    big_name = 'big_' + wb_name + '.xlsx'
    workbook = xlsxwriter.Workbook(big_name)
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    for list_ele in big_list:
        for item in list_ele:
            worksheet.write(row, column, item[0])
            column += 1
            worksheet.write(row, column, item[1])
            column += 1
        column = 0
        row += 1
    workbook.close()

    # write class and confidence per bounding box to the xlsfile
    dict_name = 'dict_' + wb_name + '.xlsx'
    # workbook = xlsxwriter.Workbook(r'dict.xlsx')
    workbook = xlsxwriter.Workbook(dict_name)
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    for dict_ele in score_class_list:
        item_list = []
        for key, value in dict_ele.items():
            item_list.append(key)
            item_list.append(value)
        for item in item_list:
            worksheet.write(row, column, item)
            column += 1
        column = 0
        row += 1
    workbook.close()

    # write final_class to the xlsfile
    ans_name = 'ans_' + wb_name + '.xlsx'
    workbook = xlsxwriter.Workbook(ans_name)
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    # write  down answer sheet
    for item in answer_sheet_list:
        # write operation perform
        worksheet.write(row, column, item)
        row += 1
    workbook.close()

    # write down confidence for every class
    integrate_name = 'integrate_' + wb_name + '.csv'
    with open(integrate_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 2 items a row
        for i in range(0, len(for_ensemble_confidence_list), 2):  # step by threes.
            writer.writerow(for_ensemble_confidence_list[i:i + 2])


    return T, F, accuracy, mb, bm