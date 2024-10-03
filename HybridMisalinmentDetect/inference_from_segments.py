import SimpleITK as sitk
import numpy as np
import segmentationmetrics as sm
import pandas as pd
from termcolor import cprint
import pickle
from tqdm import tqdm
import os
from glob import glob
from natsort import os_sorted
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import monai
import torch
import roc_utils as ru


def classification_eval_function(prediction_activated, groundtruth, 
                        weights = tuple([1 for x in range(1000)]),
                        is_binary = True,
                        ):


    confusion_metrics = monai.metrics.ConfusionMatrixMetric(metric_name=["F1",
                                                                              "sensitivity",
                                                                              "specificity",
                                                                              "precision",
                                                                              "accuracy"],
                                                                 reduction="mean_batch")
    try:
        prediction_binary = [(x == x.max()).float() for x in prediction_activated]
    except:
        prediction_binary = [(x == x.max()) for x in prediction_activated]
        prediction_binary = torch.tensor(prediction_binary)
        groundtruth = torch.tensor(groundtruth)
        prediction_activated = torch.tensor(prediction_activated)
    
    confusion_metrics(prediction_binary, groundtruth)
    if is_binary:
        F1_score,sensitivity_score,specificity_score,precision_score,accuracy_score = [float(x.detach()[0]) for x in confusion_metrics.aggregate()]
    else:
        F1_score,sensitivity_score,specificity_score,precision_score,accuracy_score = [torch.nanmean(x).item() for x in confusion_metrics.aggregate()]
    ############
    F1_score_multi_class,sensitivity_score_multi_class,specificity_score_multi_class,precision_score_multi_class,accuracy_score_multi_class = [x.cpu().numpy() for x in confusion_metrics.aggregate()]
    
    if isinstance(prediction_activated, np.ndarray):
        if prediction_activated.ndim > 1:
            num_classes = prediction_activated.shape[1]
            AUC_multi_class = []
            for index_class in range(num_classes):
                AUC_multi_class.append(ru.compute_roc(X=prediction_activated[:,index_class], y=groundtruth[:,index_class], pos_label=True).auc)
        else:
            AUC_multi_class = 0  
    else:# isinstance(prediction_activated, list):
        num_classes = len(prediction_activated[0]) 
        if len(prediction_activated[0]) > 2:
            AUC_multi_class = []
            for index_class in range(num_classes):
                AUC_multi_class.append(ru.compute_roc(X = [x[index_class].cpu() for x in prediction_activated],
                                                      y = [x[index_class].cpu() for x in groundtruth],
                                                      pos_label=True).auc)
        else:
            AUC_multi_class = 0  
    ############
    ROC_AUC_mteric = monai.metrics.ROCAUCMetric()

    ROC_AUC_mteric(prediction_activated, groundtruth)
    AUC_metric = ROC_AUC_mteric.aggregate()
    
    classification_eval = {"F1_score" : F1_score, "sensitivity_score" : sensitivity_score, 
                           "specificity_score" : specificity_score, "precision_score" : precision_score, 
                           "accuracy_score" : accuracy_score,
                             "sens_spec_average" : (sensitivity_score + specificity_score) / 2,
                               "AUC_metric" : AUC_metric,
                               }
    multi_calssification_eval = {"F1_score" : F1_score_multi_class, "sensitivity_score" : sensitivity_score_multi_class, 
                           "specificity_score" : specificity_score_multi_class, "precision_score" : precision_score_multi_class, 
                           "accuracy_score" : accuracy_score_multi_class,
                             "sens_spec_average" : (sensitivity_score_multi_class + specificity_score_multi_class) / 2,
                               "AUC_metric" : AUC_multi_class,
                               }
    if num_classes>2:
        def weigted_average(array, weights_function = weights):
            if isinstance(array, int):
                return 0
            pure_array = [x for x in array if not np.isnan(x)]
            if not len(pure_array):
                average_weighted = 0
            else:
                average_weighted = np.average(pure_array, weights = weights_function[:len(pure_array)])
            return average_weighted
        weighted_classification_eval = {"F1_score" : weigted_average(F1_score_multi_class),
                                                                     "sensitivity_score" : weigted_average(sensitivity_score_multi_class), 
                               "specificity_score" : weigted_average(specificity_score_multi_class),
                                                                    "precision_score" : weigted_average(precision_score_multi_class), 
                               "accuracy_score" : weigted_average(accuracy_score_multi_class),
                                 "sens_spec_average" : (weigted_average(sensitivity_score_multi_class) + weigted_average(specificity_score_multi_class)) / 2,
                                   "AUC_metric" : weigted_average(AUC_multi_class),
                                   }
    else:
        weighted_classification_eval = classification_eval
            
    ROC_AUC_mteric.reset()
    confusion_metrics.reset()
    return multi_calssification_eval, classification_eval, weighted_classification_eval



def ROC_Plot(groundtruth_class, predicted_probailities, output_url, dpi = 600, show = False):
    """groundtruth_class has to have num_class columns
    (y_onehot_test is train classifier code)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    import numpy as np
    
    n_classes = len(predicted_probailities[0])
    y_test = np.array([x.tolist() for x in groundtruth_class])
    y_score = np.array(predicted_probailities)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw=2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.savefig(output_url, dpi = dpi, bbox_inches = "tight")
    if show:
        plt.show()
    return plt.close()


def ConfusionMatrix(GroundTruthLabel, PrecitedLabel,
                    names = [f"class-{x}" for x in range(1000)],
                    normalize = True,
                    ouput_url = "none", dpi = 1000, show = False,
                    cmap = "afmhot",
                    ):

    number_of_calsses = len(set(GroundTruthLabel))
    names = names[:number_of_calsses]
    confusion_matrix = metrics.confusion_matrix(GroundTruthLabel, PrecitedLabel)
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = names)
    fig, ax = plt.subplots()
    im = cm_display.plot(cmap=cmap, ax=ax)
    if normalize:
        im.im_.set_clim(0, 1)
    if show:
        plt.show()
    if ouput_url != "none":
        plt.savefig(ouput_url, dpi = dpi, bbox_inches='tight' )
    return plt.close()
  
def ROC_fancy_plot(prediction_activated,
                   groundtruth,
                   bootstrap = False,
                   n_samples = 100,
                   pos_label = True,
                   output_url = "none",
                   dpi = 600,
                   single_class = "all",
                   show_opt = True,
                   colors = ("red", "blue", "green", "cyan"),
                   font_size = 10,
                   title = "ROC Curve",
                   show = True,
                   fontsize = 10,
                   ):

    if prediction_activated.ndim == 1:
       prediction_activated = np.expand_dims(prediction_activated, axis = 1)
    if groundtruth.ndim == 1:
       groundtruth = np.expand_dims(groundtruth, axis = 1)
    
    if single_class != "all":
        #creating for one class
        prediction_activated = prediction_activated[:,single_class]
        groundtruth = groundtruth[:,single_class]
        num_classes = 1
    else:    
        num_classes = prediction_activated.shape[1]
     
    if prediction_activated.ndim == 1:
       prediction_activated = np.expand_dims(prediction_activated, axis = 1)
    if groundtruth.ndim == 1:
       groundtruth = np.expand_dims(groundtruth, axis = 1)
       
    if bootstrap:
        # create bootstrap option
        _, ax3 = plt.subplots()
        for class_index in range(num_classes):
           exec(f"ROC_Strap_{class_index} = ru.plot_roc_bootstrap(X=prediction_activated[:,class_index], y=groundtruth[:,class_index], pos_label=pos_label,n_bootstrap={n_samples}, show_opt={show_opt})")           
          
        # Place the legend outside.
        ax3.legend(fontsize = fontsize)
        ax3.set_title(title);
        if output_url != "none":
            plt.savefig(output_url, dpi = dpi, bbox_inches = "tight")
        if show:
            plt.show()
    else:
        #without BootStraping
        _, ax3 = plt.subplots()
        for class_index in range(num_classes):
           exec(f"ROC_NoStrap_{class_index} = ru.compute_roc(X=prediction_activated[:,class_index], y=groundtruth[:,class_index], pos_label=pos_label)")           
           exec(f"ru.plot_roc(ROC_NoStrap_{class_index}, label='Class{class_index}', color='{colors[class_index]}', ax=ax3, show_opt={show_opt})")
          
        # Place the legend outside.
        ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax3.set_title(title);
    
    if output_url != "none":
        plt.savefig(output_url, dpi = dpi, bbox_inches = "tight")
    if show:
        plt.show()
    plt.close()
    
def classification_from_prob_df(excel_url, names = ["Low-Q", "High-Q"], is_binary = True,  save_plots = True, save_eval = True, output_folder = "same", normalize_cm = False):

    if isinstance(excel_url, str):
        df = pd.read_excel(excel_url)
    else:
        df = excel_url
    if output_folder != "same":
        os.makedirs(output_folder, exist_ok = True)
    # prediction
    predict_probs_col_names = [x for x in df.columns if "class-" in x and "-Prob" in x ]
    prediction_activated = df[predict_probs_col_names].values
    # groundtruth
    groundtruth = np.zeros([df.shape[0],2])
    for index, ground_truth_class in enumerate(df["GroundTruthLabel"]):
        groundtruth[index,int(ground_truth_class)] = 1
    if save_plots:
        ConfusionMatrix(df["GroundTruthLabel"], df["PredictedLabel"], 
                                              ouput_url = excel_url.replace(".xlsx", "confusion.png") if output_folder == "same" else os.path.join(output_folder, "confusion.png"),
                                              dpi = 600,
                                              normalize = normalize_cm,
                                              cmap = "afmhot",
                                              names = names,
                                              )
        
        
        single_class = 0
        ROC_fancy_plot(prediction_activated,
                           groundtruth,
                           bootstrap = True,
                           single_class = single_class,
                           show = False,
                           n_samples = 1000,
                           dpi = 600,
                           show_opt = False,
                           output_url = excel_url.replace(".xlsx", "ROC-fancy.png") if output_folder == "same" else os.path.join(output_folder, "ROC-fancy.png"),
                           title = "",
                           fontsize = 10,
                           # title = name.replace(".xlsx", ""),
                           )
    if save_eval:
        multi_calssification_eval, classification_eval, weighted_classification_eval = classification_eval_function(prediction_activated,
                                                                                                                                 groundtruth, 
                                                                                                                                 
                                                                            weights = tuple([1 for x in range(1000)]),
                                                                            is_binary = is_binary,
                                                                            )
        classification_eval_df = pd.DataFrame(classification_eval, index = [0])
        if output_folder == "same":
            classification_eval_df.to_excel(excel_url.replace(".xlsx", "evaluations.xlsx"))
        else:
            classification_eval_df.to_excel(os.path.join(output_folder, "evaluations.xlsx"))
            
        return classification_eval_df
    
def match_space(input_image, reference_image, interpolate = "linear", DefaultPixelValue = 0):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
  
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # Set the default pixel value to -1000
    resampler.SetDefaultPixelValue(DefaultPixelValue)
    if interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(input_image)
    return resampled_image

def segment_volume(segment, spacing = "from-segment", segment_number = 1):
    """
    Parameters
    ----------
    segment : segment URL or segment sitk image.
    spacing : TYPE, optional
        DESCRIPTION. The default is "from-segment".
    segment_number : TYPE, optional
        DESCRIPTION. The default is 1.
    Returns
    -------
    number_of_voxels : number of voxels in the semgnet
    segment_volume : segment volume in ml (1e3 mm3)
    """
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    if spacing == "from-segment":
        spacing = segment.GetSpacing()

    segment_binary = select_segment_number(segment, segment_value = segment_number)[0]
    segment_array = sitk.GetArrayFromImage(segment_binary)
    number_of_voxels =  segment_array.sum()
    segment_volume = number_of_voxels * np.prod(list(spacing)) / 1e3
    return number_of_voxels, segment_volume

def segment_stat(input_image, input_mask, segment_value = "all", prefix = "", force_match = False):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(input_mask, str):
        input_mask = sitk.ReadImage(input_mask)
    if force_match:
        input_mask = match_space(input_image = input_mask, reference_image = input_image)
        
    input_mask = sitk.Cast(input_mask, sitk.sitkUInt8)
    unique_values = np.unique(sitk.GetArrayFromImage(input_mask))[1:]
    
    image_array = sitk.GetArrayFromImage(input_image)
    segment_array = sitk.GetArrayFromImage(input_mask)
    stats = {}
    if segment_value != "all":
        unique_values = [segment_value]
        for seg_number in unique_values:
            image_segmented_array = image_array[segment_array == seg_number]
        
            stats[f'{prefix}mean'] = np.mean(image_segmented_array)
            stats[f'{prefix}min'] =  np.min(image_segmented_array)
            stats[f'{prefix}median'] =  np.median(image_segmented_array)
            stats[f'{prefix}max'] =  np.max(image_segmented_array)
            stats[f'{prefix}sum'] =  np.sum(image_segmented_array)
            stats[f'{prefix}std'] =  np.std(image_segmented_array)
            stats[f'{prefix}volume'] = segment_volume(input_mask, segment_number  = int(seg_number))[-1]
            
    else:
        for seg_number in unique_values:
            image_segmented_array = image_array[segment_array == seg_number]
        
            stats[f'{prefix}mean_SEG_{seg_number}'] = np.mean(image_segmented_array)
            stats[f'{prefix}min_SEG_{seg_number}'] =  np.min(image_segmented_array)
            stats[f'{prefix}median_SEG_{seg_number}'] =  np.median(image_segmented_array)
            stats[f'{prefix}max_SEG_{seg_number}'] =  np.max(image_segmented_array)
            stats[f'{prefix}sum_SEG_{seg_number}'] =  np.sum(image_segmented_array)
            stats[f'{prefix}std_SEG_{seg_number}'] =  np.std(image_segmented_array)
            stats[f'{prefix}volume_SEG_{seg_number}'] = segment_volume(input_mask, segment_number  = int(seg_number))[-1]
    return stats

def segment_match_eval(reference_url, predicted_url, pixdim = (1,1,1)):
    
    if isinstance(reference_url, str):
        reference_array = sitk.GetArrayFromImage(sitk.ReadImage(reference_url, sitk.sitkUInt8))
        pixdim = sitk.ReadImage(reference_url).GetSpacing()
    elif isinstance(reference_url, sitk.Image):
        reference_array = sitk.GetArrayFromImage(reference_url)
        pixdim = reference_url.GetSpacing()
    else:
        reference_array = reference_url
        
        
    if isinstance(predicted_url, str):
        predicted_array = sitk.GetArrayFromImage(sitk.ReadImage(predicted_url, sitk.sitkUInt8))
    elif isinstance(predicted_url, sitk.Image):
        predicted_array = sitk.GetArrayFromImage(predicted_url)
    else:
        predicted_array = predicted_url
       
    whole_info_df = pd.DataFrame()
    unique_segments = np.unique(reference_array)
    if len(unique_segments) == 1:
       cprint("empty segmentation, emty df returned", "white", "on_red") 
       return whole_info_df
    for seg_number in unique_segments[1:]:
        predicted_array_unique = predicted_array
        predicted_array_unique = predicted_array == seg_number
        reference_array_unique = reference_array == seg_number
        metrics_df = sm.SegmentationMetrics(predicted_array_unique, reference_array_unique, pixdim).get_df()
        metrics_df.reset_index(drop=True, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        metrics_df = metrics_df.T
        metrics_df["Segment_number_to_save"] = seg_number
        column_names = metrics_df.iloc[0]
        metrics_df = metrics_df.drop("Metric")
        metrics_df["Segment_number"] = seg_number
        metrics_df = metrics_df.set_index('Segment_number')
        whole_info_df = pd.concat([whole_info_df, metrics_df], axis = 0, ignore_index=True)
    column_names = column_names.rename("Segment-Number")
    
    whole_info_df.columns = column_names
    whole_info_df = whole_info_df.rename(columns = {max(unique_segments):'actual_segment_number'})
    return whole_info_df

def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage

def select_segment_number(segment, segment_value):
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    segment_array = sitk.GetArrayFromImage(segment)
    segment_array[segment_array != segment_value] = 0
    segment_array[segment_array == segment_value] = 1
    segment_single = sitk.GetImageFromArray(segment_array)
    segment_single = CopyInfo(segment, segment_single)
    return segment_single, segment_array

def TableUnify(list_excels, target_url = "none", allow_tqdm = True, skip_error = False, sheet_name = 0):
    import pandas as pd
    from tqdm import tqdm
    import os
    import time
    file_name, file_extension = os.path.splitext(list_excels[0])
    if file_extension == ".csv":
        concatenated_data = pd.read_csv(list_excels[0], decimal='.', encoding='utf-8')
        concatenated_data["excel_url"] = [list_excels[0]] * concatenated_data.shape[0]
        concatenated_data["excel_name"] = [os.path.basename(list_excels[0])] * concatenated_data.shape[0]
        concatenated_data = concatenated_data[list(concatenated_data.columns[-2:]) + list(concatenated_data.columns[:-2])]
    elif file_extension == ".xlsx":
        concatenated_data = pd.read_excel(list_excels[0], sheet_name = sheet_name)
        concatenated_data["excel_url"] = [list_excels[0]] * concatenated_data.shape[0]
        concatenated_data["excel_name"] = [os.path.basename(list_excels[0])] * concatenated_data.shape[0]
        concatenated_data = concatenated_data[list(concatenated_data.columns[-2:]) + list(concatenated_data.columns[:-2])]
        
    list_excels = list_excels[1:]
    if skip_error:
        if allow_tqdm:
            for url in tqdm(list_excels, desc = "Table Unify "):
                # time.sleep(.1)
                try:
                    file_name, file_extension = os.path.splitext(url)
                    if file_extension == ".csv":
                        df = pd.read_csv(url)
                        df["excel_url"] = [url] * df.shape[0]
                        df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                        df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                    elif file_extension == ".xlsx":
                        df = pd.read_excel(url, sheet_name = sheet_name)
                        df["excel_url"] = [url] * df.shape[0]
                        df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                        df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                    concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
                except:
                    print(url)

        else:
            for url in list_excels:
                try:
                    file_name, file_extension = os.path.splitext(url)
                    if file_extension == ".csv":
                        df = pd.read_csv(url)
                        df["excel_url"] = [url] * df.shape[0]
                        df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                        df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                    elif file_extension == ".xlsx":
                        df = pd.read_excel(url, sheet_name = sheet_name)
                        df["excel_url"] = [url] * df.shape[0]
                        df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                        df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                    concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
                except:
                    pass
                
        if target_url != "none":
            concatenated_data.to_excel(target_url, index=False)
            
    else:    
        if allow_tqdm:
            for url in tqdm(list_excels, desc = "Table Unify "):
                # time.sleep(.1)
                file_name, file_extension = os.path.splitext(url)
                if file_extension == ".csv":
                    df = pd.read_csv(url)
                    df["excel_url"] = [url] * df.shape[0]
                    df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                    df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                elif file_extension == ".xlsx":
                    df = pd.read_excel(url, sheet_name = sheet_name)
                    df["excel_url"] = [url] * df.shape[0]
                    df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                    df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
        else:
            for url in list_excels:
                file_name, file_extension = os.path.splitext(url)
                if file_extension == ".csv":
                    df = pd.read_csv(url)
                    df["excel_url"] = [url] * df.shape[0]
                    df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                    df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                elif file_extension == ".xlsx":
                    df = pd.read_excel(url, sheet_name = sheet_name)
                    df["excel_url"] = [url] * df.shape[0]
                    df["excel_name"] = [os.path.basename(url)] * df.shape[0]
                    df = df[list(df.columns[-2:]) + list(df.columns[:-2])]
                concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
        if target_url != "none":
            concatenated_data.to_excel(target_url, index=False)
    return concatenated_data

   
def predict_RF_ensemble(ensemble_model_urls, 
                        predictors, labels = "none",
                        weights = (1,) * 100, 
                        names = ["No-RMA", "RMA"], 
                        output_folder = "none", 
                        normalize_cm = True):

    ensemble_models = []
    for model_url in tqdm(ensemble_model_urls, desc = "loading models", colour="red", ncols = 100):
        with open(model_url, 'rb') as file:
            loaded_model = pickle.load(file)
        ensemble_models.append(loaded_model)
    # ensemble_models = [joblib.load(model_url) for model_url in ensemble_model_urls]
    cprint("Predicting .... ", "white", "on_cyan")
    predictions = [model.predict(predictors) for model in ensemble_models]
    probabilities = [model.predict_proba(predictors) for model in ensemble_models]
    num_calsses = probabilities[0].shape[1]
    
    for fold_index in range(len(ensemble_model_urls)):
        prediction_this_fold = predictions[fold_index]
        probabnilites_this_fold = probabilities[fold_index]
        evaluation_this_fold_df = pd.DataFrame(probabnilites_this_fold, columns = ["class-0-Prob", "class-1-Prob"])
        evaluation_this_fold_df["GroundTruthLabel"] = labels.values
        evaluation_this_fold_df["PredictedLabel"] = prediction_this_fold
        if not isinstance(labels, str) and (labels != "none").any() and not output_folder == "none":
            classification_from_prob_df(excel_url = evaluation_this_fold_df,
                                                              names = names, 
                                                              output_folder = os.path.join(output_folder, f"Fold--{fold_index}"),
                                                              normalize_cm = normalize_cm,
                                                              )
            
    if not isinstance(labels, str) and (labels != "none").any() and not output_folder == "none":
        TableUnify(list_excels = os_sorted(glob(os.path.join(output_folder, "Fold--*", "evaluations.xlsx" ))),
                                target_url = os.path.join(output_folder, "All-Fold-evaluations.xlsx")) 
                  

    sum_images_probabilities_weighted = sum(probabilities)
    # this is the weights in the inference time
    for class_number in range(num_calsses):
        sum_images_probabilities_weighted[:, 1] = sum_images_probabilities_weighted[:, 1] * weights[class_number]
        
        
        
    predictions_from_probabilities = np.argmax(sum_images_probabilities_weighted, axis=1)
    final_prediction = pd.DataFrame(predictions).mode(axis=0).iloc[0]
    
    evaluation_non_weighted_df = pd.DataFrame(sum_images_probabilities_weighted / len(ensemble_model_urls), columns = ["class-0-Prob", "class-1-Prob"])
    evaluation_non_weighted_df["GroundTruthLabel"] = labels.values
    evaluation_non_weighted_df["PredictedLabel"] = final_prediction
    
    
    if not isinstance(labels, str) and (labels != "none").any() and not output_folder == "none":
        classification_from_prob_df(excel_url = evaluation_non_weighted_df,
                                                          names = names, 
                                                          output_folder = output_folder,
                                                          normalize_cm = normalize_cm,
                                                          )
        evaluation_non_weighted_df.to_excel(os.path.join(output_folder, "probabilities.xlsx"))
        
    normal_predictions = {}
    normal_predictions["predicted_labels"] = final_prediction 
    normal_predictions["predicted_label_folds"] = predictions 
    
   
    weighted_predictions = {}
    weighted_predictions["probabilities"] = probabilities 
    weighted_predictions["sum_probalaties_weighted"] = sum_images_probabilities_weighted 
    weighted_predictions["predicted_labels_weighted"] = predictions_from_probabilities 
    
    cprint("Completed .... ", "white", "on_green")

    return normal_predictions, weighted_predictions

def petct_ram_detect(
        model_directory, # the fodler containing .pkl files downloaded
        list_of_casenames: list,
        list_of_pet_segmentations: list,
        list_of_ct_segmentations: list, 
        refernce_labels: list, # the reference labels if you have them
        results_folder:str,
        segment_values = {
            "Liver": 1,
            "Spleen": 2,
            "Lungs": 3,
            "Heart": 4,
            },
        ):
    
    # calcualte inputs
    predictors_df = pd.DataFrame()
    counter = 0
    segment_pet = r"C:\yzdn\explain-nnunet-black-box\epoch--0000\CT000001-nncrop--LiverExplain--yzdnn.nii.gz"
    segment_ct = r"C:\yzdn\explain-nnunet-black-box\epoch--0003\CT000001-nncrop--LiverExplain--yzdnn.nii.gz"
    for segment_pet, segment_ct, case_name in tqdm(zip(list_of_pet_segmentations, list_of_ct_segmentations, list_of_casenames),
                                        total = len(list_of_ct_segmentations), 
                                        desc = "extracting input metrics"):
        
        segment_liver_ct = select_segment_number(segment_ct, segment_value = segment_values["Liver"])[0]
        segment_liver_nc = select_segment_number(segment_pet, segment_value = segment_values["Liver"])[0]
        
        segment_spleen_ct = select_segment_number(segment_ct, segment_value = segment_values["Spleen"])[0]
        segment_spleen_nc = select_segment_number(segment_pet, segment_value = segment_values["Spleen"])[0]
              
        segment_lungs_ct = select_segment_number(segment_ct, segment_value = segment_values["Lungs"])[0]
        segment_lungs_nc = select_segment_number(segment_pet, segment_value = segment_values["Lungs"])[0]
        
        segment_heart_ct = select_segment_number(segment_ct, segment_value = segment_values["Heart"])[0]
        segment_heart_nc = select_segment_number(segment_pet, segment_value = segment_values["Heart"])[0]
                
        segment_liver_eval = segment_match_eval(segment_liver_ct, match_space(input_image = segment_liver_nc, reference_image = segment_liver_ct))
        segment_lungs_eval = segment_match_eval(segment_lungs_ct, match_space(input_image = segment_lungs_nc, reference_image = segment_lungs_ct))
        segment_heart_eval = segment_match_eval(segment_heart_ct, match_space(input_image = segment_heart_nc, reference_image = segment_heart_ct))
        
        # lung-CT liver-NC overlap
        segment_liver_nc = match_space(input_image = segment_liver_nc, reference_image = segment_liver_ct)
        lung_liver_overlap_segment = segment_lungs_ct * 0
        lung_liver_overlap_segment[(segment_liver_nc == 1) & (segment_lungs_ct == 1)] = 10
        
        segment_liver_overlap_eval = segment_stat(lung_liver_overlap_segment, lung_liver_overlap_segment)
    
        # lung-CT spleen-NC overlap
        segment_spleen_nc = match_space(input_image = segment_spleen_nc, reference_image = segment_spleen_ct)
        lung_spleen_overlap_segment = segment_lungs_ct * 0
        lung_spleen_overlap_segment[(segment_spleen_nc == 1) & (segment_lungs_ct == 1)] = 10
        
        segment_spleen_overlap_eval = segment_stat(lung_spleen_overlap_segment, lung_spleen_overlap_segment)
      
        predictors_df.at[counter, "casename"] = case_name
        predictors_df.at[counter, "Dice--Liver"] = segment_liver_eval["Dice"]
        predictors_df.at[counter, "Jaccard--Liver"] = segment_liver_eval["Jaccard"]
        predictors_df.at[counter, "Volume Difference--Liver"] = segment_liver_eval["Volume Difference"]
        predictors_df.at[counter, "Mean Surface Distance--Liver"] = segment_liver_eval["Mean Surface Distance"]
        
        predictors_df.at[counter, "Dice--lungs"] = segment_lungs_eval["Dice"]
        predictors_df.at[counter, "Jaccard--lungs"] = segment_lungs_eval["Dice"]
        predictors_df.at[counter, "Volume Difference--lungs"] = segment_lungs_eval["Volume Difference"]
        
        predictors_df.at[counter, "Lung-liver-overlap"] = segment_liver_overlap_eval["volume_SEG_10"]
        predictors_df.at[counter, "Lung-spleen-overlap"] = segment_spleen_overlap_eval["volume_SEG_1"]
        
        predictors_df.at[counter, "Jaccard--Heart"] = segment_heart_eval["Jaccard"]
        
        counter =+ 1
        
        ## predict class
        ensemble_model_urls = os_sorted(glob(os.path.join(model_directory, "RF_Explainable-Motion-Detction-all-included_Fold_*.pkl")))
        non_wighted_predictionss, _ = predict_RF_ensemble(ensemble_model_urls,
                                                          predictors = predictors_df,
                                                          labels = refernce_labels, 
                                                          output_folder = results_folder,
                                                          )
        
        return non_wighted_predictionss
        
