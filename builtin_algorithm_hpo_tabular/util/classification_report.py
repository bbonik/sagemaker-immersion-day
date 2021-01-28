    # -*- coding: utf-8 -*-
"""
@author: vasileios vonikakis
@email: bbonik@gmail.com
Functions to plot visually pleasing reports for machine learning models.
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import collections



def plot_confusion_matrix(
        confusion_matrix,            
        class_names_list=['Class1', 'Class2'],               
        axis=None,           
        title='Confusion matrix',               
        label_rotation_true=0,             
        label_rotation_predict=0,              
        plot_style='seaborn',
        colormap=plt.cm.Blues):
    
    '''
    --------------------------------------------------------------------------
             Plots a nice comfusion matrix for a classification model 
    --------------------------------------------------------------------------
    Generates a comfucion matrix, either alone, or embeded in another figure.
    
    INPUTS
    ------
    confusion_matrix: numpy array
        The actual confusion matrix to be plotted. 
    class_names_list: list of strings
        Names of the 2 classes. If None, then Class0, Class1... etc. is used. 
    axis: matplotlib axis
        If None, a new standalone drawing is created. If an axis is passes, 
        then the graph is embeded inside the axis of the other figure. 
    title: string
        Title of the graph
    label_rotation_true: number in degrees
        Rotation to be applied for the actual (true) label classes.
    label_rotation_predict: number in degrees
        Rotation to be applied for the predicted label classes.
    plot_style: string
        Plotting style to be used. 
    colormap: matplotlib colormap
        Colormap to be used for plotting 
    '''
    
    
    if axis is None:  # for standalone plot
        plt.figure()
        ax = plt.gca()
    else:  # for plots inside a subplot
        ax = axis
        
    plt.style.use(plot_style)
    
    # normalizing matrix to [0,100%]
    confusion_matrix_norm = (confusion_matrix.astype('float') / 
                             confusion_matrix.sum(axis=1)[:, np.newaxis])
    confusion_matrix_norm = np.round(100 * confusion_matrix_norm, 2)

    ax.imshow(confusion_matrix_norm,
               interpolation='nearest',
               cmap=colormap,
               vmin=0,  # to make sure colors are scaled between [0,100%]
               vmax=100)
 
    ax.set_title(title)
    tick_marks = np.arange(len(class_names_list))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names_list, rotation=label_rotation_predict)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names_list, rotation=label_rotation_true)
    
    for i, j in itertools.product(range(confusion_matrix.shape[0]), 
                                  range(confusion_matrix.shape[1])):
        ax.text(j, i, 
                 (str(confusion_matrix[i, j]) + '\n(' + 
                  str(confusion_matrix_norm[i,j]) + '%)'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_norm[i, j] >50 else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.grid(False)
    
    if axis is None:  # for standalone plots
        plt.tight_layout()
        plt.show()






def plot_precision_recall_curve(
        y_actual,                     
        y_predict,                     
        axis=None,                 
        plot_style='seaborn'):
    
    '''
    --------------------------------------------------------------------------
         Plots a precision-recall curve for a binary classification model 
    --------------------------------------------------------------------------
    Generates a precision-recall curve, either alone, or embeded insode 
    another figure.
    
    INPUTS
    ------
    y_actual: numpy array (N,)
        Actual ground truth values.
    y_predict_proba: numpy array (N,)
        Predicted probabilities, as derived from the binary classifier.
    axis: matplotlib axis
        If None, a new standalone drawing is created. If an axis is passes, 
        then the graph is embeded inside the axis of the other figure. 
    plot_style: string
        Plotting style to be used. 
    '''
    
    # get metrics
    metrics_P, metrics_R, _ = metrics.precision_recall_curve(
        y_actual, 
        y_predict
        )
    metrics_AP = metrics.average_precision_score(
        y_actual, 
        y_predict
        )
    
    if axis is None:  # for standalone plot
        plt.figure()
        ax = plt.gca()
    else:  # for plots inside a subplot
        ax = axis
        
    plt.style.use(plot_style)
    ax.set_aspect(aspect=0.95)
    ax.step(metrics_R, metrics_P, color='b', where='post', linewidth=0.7)
    ax.fill_between(metrics_R, metrics_P, step='post', alpha=0.2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.set_title('Precision-Recall curve: AP={0:0.3f}'.format(metrics_AP))
    
    if axis is None:  # for standalone plots
        plt.tight_layout()
        plt.show()





def plot_roc_curve(
        y_actual,
        y_predict,
        axis=None,
        plot_style='seaborn'):
    
    '''
    --------------------------------------------------------------------------
               Plots a ROC curve for a binary classification model 
    --------------------------------------------------------------------------
    Generates a ROC curve, either alone, or embeded insode another figure.
    
    INPUTS
    ------
    y_actual: numpy array (N,)
        Actual ground truth values.
    y_predict_proba: numpy array (N,)
        Predicted probabilities, as derived from the binary classifier.
    axis: matplotlib axis
        If None, a new standalone drawing is created. If an axis is passes, 
        then the graph is embeded inside the axis of the other figure. 
    plot_style: string
        Plotting style to be used. 
    '''

    
    # get metrics
    metrics_FPR, metrics_TPR, _ = metrics.roc_curve(y_actual, y_predict)
    metrics_AUC = metrics.roc_auc_score(y_actual, y_predict)
    
    
    if axis is None:  # for standalone plot
        plt.figure()
        ax = plt.gca()
    else:  # for plots inside a subplot
        ax = axis
    
    plt.style.use(plot_style)
    ax.set_aspect(aspect=0.95)
    ax.plot(metrics_FPR, metrics_TPR, color='b', linewidth=0.7)
    ax.fill_between(metrics_FPR, metrics_TPR, step='post', alpha=0.2,color='b')
    ax.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve: AUC={0:0.3f}'.format(metrics_AUC))
    
    if axis is None:  # for standalone plots
        plt.tight_layout()
        plt.show()




def plot_text(text, font_size=10, axis=None):
    
    '''
    --------------------------------------------------------------------------
                          Plots text inside an image
    --------------------------------------------------------------------------
    
    INPUTS
    ------
    text: string
        Actual text to be included inside an image.
    font_size: int 
        Font size of the text.
    axis: matplotlib axis
        If None, a new standalone drawing is created. If an axis is passes, 
        then the graph is embeded inside the axis of the other figure.
    '''
    
    plt.rcParams.update({'font.size': font_size})
    
    if axis is None:  # for standalone plot
        plt.figure()
        ax = plt.gca()
    else:  # for plots inside a subplot
        ax = axis
        
    # set background white
    ax.set_axis_off()
    ax.set_frame_on(True)
    ax.grid(False)
    
    ax.text(
        x=0.8,
        y=0, 
        s=text,
        horizontalalignment="right",
        color="black"
        )






    
    
def estimate_best_threshold(y_actual, y_predict_proba):
    
    '''
    --------------------------------------------------------------------------
                  Find the best threshold for a binary classifier
    --------------------------------------------------------------------------
    Estimates the threshold, for which, the maximum F1 score is achieved. 
    Note : y_actual and y_predict_proba should have the same size. 
    
    INPUTS
    ------
    y_actual: numpy array (N,)
        Actual ground truth values.
    y_predict_proba: numpy array (N,)
        Predicted probabilities, as derived from the binary classifier.
    
    OUTPUTS
    ------
    best_threshold: float
        Threshold at which, the highest F1 score is achieved.
    '''
    
    GRANULARITY = 200
    DECIMALS = 3
    
    best_threshold = None
    
    ls_f1 = []
    ls_thresholds = []
        
    for decision_threshold in np.linspace(0, 1, GRANULARITY+1):
        # threshold probability
        y_decision = y_predict_proba.copy()
        y_decision[y_decision>decision_threshold] = 1
        y_decision[y_decision<1] = 0
        y_decision = y_decision.astype(bool)
        
        f1 = round(metrics.f1_score(y_actual, y_decision), DECIMALS)
        ls_f1.append(f1)
        ls_thresholds.append(decision_threshold)
        
    array_f1 = np.array(ls_f1)
    max_f1 = array_f1.max()
    indx = array_f1 == max_f1  # keep the indices of the maximum f1
    array_thr = np.array(ls_thresholds)
    best_threshold = np.median(array_thr[indx])  # get the median among maxs
    
    return round(best_threshold, 3)






def generate_threshold_report(
        y_actual, 
        y_predict_proba, 
        tp_decision_thresholds=(0.1, 0.25, 0.5, 0.75, 0.9), 
        class_names_list=None,
        plot_style='seaborn'):
    
    '''
    --------------------------------------------------------------------------
         Generates a threshold analysis for a binary classification model
    --------------------------------------------------------------------------
    Generates multiple confusion matrices for different decision thresholds,
    as well as graphs for Precision/Recall/F1score and FPR/FNR/TPR/TNR
    Note : y_actual and y_predict_proba should have the same size. 
    
    INPUTS
    ------
    y_actual: numpy array (N,)
        Actual ground truth values.
    y_predict_proba: numpy array (N,)
        Predicted probabilities, as derived from the binary classifier.
    tp_decision_thresholds: tuple of thresholds
        Thresholds for which, confusion matrices will be generated. Thresholds 
        should be real numbers within the [0,1] interval. 
    class_names_list: list of strings
        Names of the 2 classes. If None, then Class0 and Class1 is used. 
    plot_style: string
        Plotting style to be used. 
    '''

    
    BLOCK_SIZE = 3
    HEIGHT = 5
    total_graphs = len(tp_decision_thresholds)

    # find out how many classes we have in the test set
    number_of_classes = len(np.unique(y_actual))
    
    # if names are not provided for classes, create some
    if class_names_list is None:
        class_names_list = ['Class'+str(i) for i in range(number_of_classes)]

    # TODO!!!!!!
    if len(y_predict_proba.shape) > 1:
        y_predict_proba = y_predict_proba[:,1]  # keep probabilities of class1
    
    # setting up figure layout
    plt.style.use(plot_style)
    fig = plt.figure(
        constrained_layout=False, 
        figsize=(total_graphs * BLOCK_SIZE, HEIGHT * 2)
        )
    grid = fig.add_gridspec(2, 1)  # 2 lines
    fig_up = grid[0].subgridspec(2, total_graphs)  # 1st 2 x total_graphs
    fig_down = grid[1].subgridspec(1, 2)  # 2nd line 2 grpahs
    
    #---- generating multiple confusion matrices
    for i,decision_threshold in enumerate(tp_decision_thresholds):
        
        # threshold probability
        y_decision = y_predict_proba.copy()
        y_decision[y_decision>decision_threshold] = 1
        y_decision[y_decision<1] = 0
        y_decision = y_decision.astype(bool)
        
        # generate classification report
        title = 'Decision threshold: ' + str(decision_threshold)
        metrics_report = metrics.classification_report(
            y_actual, 
            y_decision, 
            target_names=class_names_list
            )
        metrics_ACC = metrics.accuracy_score(y_actual, y_decision)
        metrics_report += ('\n Total accuracy = ' + 
                           str(round(metrics_ACC*100,3)) + '%')
        metrics_report += '\n\n\n'
        
        # plot matrices
        ax = fig.add_subplot(fig_up[0, i])
        plot_confusion_matrix(
            metrics.confusion_matrix(y_actual, y_decision),
            class_names_list=class_names_list, 
            axis=ax,
            title=title,
            label_rotation_true=90,
            label_rotation_predict=0,
            )
        
        # plot repots
        ax = fig.add_subplot(fig_up[1, i])
        plot_text(metrics_report, font_size=8, axis=ax)
    

    #---- generating PR/RE/F1, FPR/FNR/TPR/TNR graphs
    ls_scanned_thresholds = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_fp = []
    ls_fn = []
    ls_tp = []
    ls_tn = []
    
    for decision_threshold in np.linspace(0,1,201):
        
        # threshold probability
        y_decision = y_predict_proba.copy()
        y_decision[y_decision>decision_threshold] = 1
        y_decision[y_decision<1] = 0
        y_decision = y_decision.astype(bool)
        
        confusion_matrix = metrics.confusion_matrix(y_actual, y_decision)
        confusion_matrix_norm = (confusion_matrix.astype('float') / 
                                 confusion_matrix.sum(axis=1)[:, np.newaxis])
        precision = round(metrics.precision_score(y_actual, y_decision), 3)
        recall = round(metrics.recall_score(y_actual, y_decision), 3)
        f1 = round(metrics.f1_score(y_actual, y_decision), 3)
        
        ls_scanned_thresholds.append(decision_threshold)
        ls_fp.append(confusion_matrix_norm[0,1]*100)
        ls_fn.append(confusion_matrix_norm[1,0]*100)
        ls_tn.append(confusion_matrix_norm[0,0]*100)
        ls_tp.append(confusion_matrix_norm[1,1]*100)
        ls_precision.append(precision)
        ls_recall.append(recall)
        ls_f1.append(f1)
        
    ax = fig.add_subplot(fig_down[0, 0])
    ax.plot(ls_scanned_thresholds,ls_precision, label='Precision', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_precision, step='mid', alpha=0.2)
    ax.plot(ls_scanned_thresholds, ls_recall, label='Recall', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_recall, step='mid', alpha=0.2)
    ax.plot(ls_scanned_thresholds,ls_f1,color='k',label='F1 score',linewidth=1)
    ax.set_xlabel('Decision threshold')
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_title('Model performance vs decision threshold')
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.21))

    ax = fig.add_subplot(fig_down[0, 1])
    ax.plot(ls_scanned_thresholds, ls_fp, label='FPR', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_fp, step='mid', alpha=0.2)
    ax.plot(ls_scanned_thresholds, ls_fn, label='FNR', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_fn, step='mid', alpha=0.2)
    ax.plot(ls_scanned_thresholds, ls_tp, label='TPR', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_tp, step='mid', alpha=0.2)
    ax.plot(ls_scanned_thresholds, ls_tn, label='TNR', linewidth=1)
    ax.fill_between(ls_scanned_thresholds, ls_tn, step='mid', alpha=0.2)
    ax.set_xlabel('Decision threshold')
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_title('FPR, FNR, TPR, TNR vs decision threshold')
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5,-0.21))

    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    # fig.suptitle('Threshold report')
    plt.show()
    
    






   
def generate_classification_report(
        y_actual, 
        y_predict_proba, 
        decision_threshold=0.5, 
        class_names_list=None,
        plot_style='seaborn',
        model_info=None):
    
    '''
    --------------------------------------------------------------------------
              Generates a detailed report for a classification model
    --------------------------------------------------------------------------
    Generates a detailed report for a classification model.
    Note : y_actual and y_predict_proba should have the same size. 
    
    INPUTS
    ------
    y_actual: numpy array (N,)
        Actual ground truth values.
    y_predict_proba: numpy array (N,)
        Predicted probabilities, as derived from the binary classifier.
    decision_threshold: real number in the interval [0,1] or None
        Threshold according which, the probabilities will be interpreted. For
        a multiclass model, decision_threshold is bypassed, and the maximum
        probability is selected as the final class. If None, then the 
        threshold with the maxmimum F1 score is selected. 
    class_names_list: list of strings
        Names of the 2 classes. If None, then Class0, Class1... etc. is used. 
    plot_style: string
        Plotting style to be used. 
    model_info: string
        Additional information about the model, to be included in the model
        summary, e.g. "Random Forest with 50 trees".
    '''
    
    

    plt.style.use(plot_style)  # set plotting style
    
    # find out how many classes we have in the test set
    number_of_classes = len(np.unique(y_actual))
    
    # if names are not provided for classes, create some
    if class_names_list is None:
        class_names_list = ['Class '+str(i) for i in range(number_of_classes)]
        
    # if model info not provided, generate something
    if model_info is None: 
        model_info = 'No model info provided'
    
    # if decision_threshold is None find one with maximum F1 score 
    if decision_threshold is None:
        decision_threshold = estimate_best_threshold(y_actual, y_predict_proba)
        
    # prepare input probabilities
    if number_of_classes <= 2:
        if len(y_predict_proba.shape) == 2:
            y_predict_proba = y_predict_proba[:,1]  # keep positive class only
        # threshold probabilities 
        y_decision = y_predict_proba.copy()
        y_decision[y_decision > decision_threshold] = 1
        y_decision[y_decision < 1] = 0
        y_decision = y_decision.astype(bool)
    else:  # mutliclass case
        y_decision = np.argmax(y_predict_proba,axis=1)  # get the maximum prob
        
    # generate ML text report
    ml_report = '----------------------- Model info ------------------------\n'  
    ml_report += model_info
    ml_report += '\n\n'         
    ml_report += ('----------------- Performance (thr=' + 
                  str(decision_threshold) + ') ----------------\n\n')

    # get initial classification report and add more text in it
    metrics_report = metrics.classification_report(
            y_actual, 
            y_decision, 
            target_names=class_names_list
            )
    metrics_ACC = metrics.accuracy_score(y_actual, y_decision)
    metrics_report += ('\n Total accuracy = ' + 
                       str(round(metrics_ACC*100,3)) + '%')
    ml_report += metrics_report
    ml_report += '\n\n\n\n\n'
        
    # generate graphs
    if number_of_classes == 2:
        fig, ax = plt.subplots(2, 2, figsize=(12,9))
        plot_text(ml_report, axis=ax[0,0])
        plot_confusion_matrix(
                metrics.confusion_matrix(y_actual, y_decision),
                class_names_list=class_names_list, 
                axis=ax[0,1]
                )
        plot_precision_recall_curve(
                y_actual, 
                y_predict_proba, 
                axis=ax[1,0]
                )
        plot_roc_curve(
                y_actual, 
                y_predict_proba, 
                axis=ax[1,1]
                )
        
    else: # single class or multi-class classification (no ROC or PR)
        fig, ax = plt.subplots(1, 2, figsize=(12,4.5))
        plot_text(ml_report, axis=ax[0])
        plot_confusion_matrix(
                metrics.confusion_matrix(y_actual, y_decision), 
                class_names_list=class_names_list, 
                axis=ax[1]
                )

    fig.suptitle('Model report', fontsize=20)
    fig.tight_layout()