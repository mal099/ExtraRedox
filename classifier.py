import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics, decomposition
from sklearn.model_selection import GridSearchCV
from numpy import interp
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFromModel, SelectKBest
from boruta import BorutaPy
from sklearn.linear_model import BayesianRidge
import os
import random
import gc
import sys
import argparse
import datetime
os.environ["PATH"] += os.pathsep + 'C:/Users/Marcus/Downloads/graphviz-2.44.1-win32/Graphviz/bin'
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
# Import imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

# Argument parser
classifier_parser = argparse.ArgumentParser(description='Classify redox proteins')

algo_args = classifier_parser.add_mutually_exclusive_group(required=True)
algo_args.add_argument('-rf', '--randomforest', action='store_true', help='classifier: Random forest')
algo_args.add_argument('-et', '--extratrees', action='store_true', help='classifier: Extra trees')
algo_args.add_argument('-svm', '--supportvector', action='store_true', help='classifier: Support vector machine')
algo_args.add_argument('-ada', '--adaboost', action='store_true', help='classifier: Adaboost')
algo_args.add_argument('-gra', '--gradientboost', action='store_true', help='classifier: Gradient Boost')

feat_args = classifier_parser.add_mutually_exclusive_group(required=True)
feat_args.add_argument('-sm', '--selectmodel', action='store_true', help='Feature selection: Select from model')
feat_args.add_argument('-sk', '--selectkbest', action='store_true', help='Feature selection: Select K best')
feat_args.add_argument('-bo', '--boruta', action='store_true', help='Feature selection: Boruta')
feat_args.add_argument('-pca', '--principal', action='store_true', help='Feature selection: Principal component analysis')

classifier_parser.add_argument('-i', '--input',
                       metavar='input name',
                       action='store',
                       type=str,
                       help='Name of input file')
classifier_parser.add_argument('-inv', '--invest',
                      metavar='investigation name',
                      action='store',
                      type=str,
                      help='Name of investigation file')
classifier_parser.add_argument('-if', '--impfile',
                      metavar='imputation name',
                      action='store',
                      type=str,
                      help='Name of imputation file')
classifier_parser.add_argument('-o', '--output',
                      metavar='output path',
                      action='store',
                      type=str,
                      help='Path to output files')
classifier_parser.add_argument('-imp',
                       '--impute',
                       action='store_true',
                       help='enable imputation')
classifier_parser.add_argument('-gs',
                       '--gridsearch',
                       action='store_true',
                       help='parameter grid search')
classifier_parser.add_argument('-hm',
                       '--heatmap',
                       action='store_true',
                       help='feature heatmap')
args = classifier_parser.parse_args()

input_path = args.input
inv_path = args.invest
imp_path = args.impfile
output_path = args.output
if not input_path:
    input_path = 'redoxprot_data_both'
    #input_path = 'devpred_output_seq_thesis'
if not inv_path:
    inv_path = 'redoxprot_data_flava'
    #inv_path = 'devpred_seq_CI'
if not imp_path:
    imp_path = 'devpred_imp_thesis'
    #imp_path = 'devpred_imp_thesis'
if not output_path:
    output_path = 'correctedCI'

if args.randomforest:
    classifier_name = 'rf'
elif args.extratrees:
    classifier_name = 'et'
elif args.supportvector:
    classifier_name = 'svm'
elif args.adaboost:
    classifier_name = 'ada'
elif args.gradientboost:
    classifier_name = 'gra'

if args.selectmodel:
    feat_sel_name = 'sm'
elif args.selectkbest:
    feat_sel_name = 'sk'
elif args.boruta:
    feat_sel_name = 'bo'
elif args.principal:
    feat_sel_name = 'pca'

if args.impute:
    imp_name = 'imp'
else:
    imp_name = 'noimp'

now = datetime.datetime.now()
if not os.path.exists('class/' + output_path):
    os.mkdir('class/' + output_path)
outDir = now.strftime(output_path + '/%Y-%m-%d_%H-%M' + classifier_name + '_' + feat_sel_name + '_flavaCTD_' + imp_name)
if not os.path.exists('class/' + outDir):
    os.mkdir('class/' + outDir)

outFile = open('class/' + outDir + '/out.txt', 'w')
resultFile = open('class/' + outDir + '/results.txt', 'w')
predFile = open('class/' + outDir + '/preds.txt', 'w')
impute_bool = args.impute
all_preds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
invest_tprs = []
invest_aucs = []
invest_mean_fpr = np.linspace(0, 1, 100)

for count in range(10):
    TP_sum = 0.0
    TN_sum = 0.0
    FP_sum = 0.0
    FN_sum = 0.0
    for count2 in range(1):
        # read dataset
        inName = '../data/' + input_path + '.txt'
        investigationName = '../data/' + inv_path + '.txt'
        incompName = '../data/' + imp_path + '.txt'
        inFile = open(inName, 'r')
        investigationFile = open(investigationName, 'r')
        outFile.write(inName + '\n' + investigationName + '\n\n')
        predFile.write('\n' + str(count) + '\n')

        # split dataset in labels and features
        X = []
        y = []
        names = []
        X_ic = []
        y_ic = []
        names_ic = []
        X_invest = []
        y_invest = []
        names_invest = []
        x_pos = 0.0
        x_neg = 0.0

        for line in inFile:
            if line[0] != '#':
                if random.random() > 0.0:#remove data to see if results get worse
                    if line[0] == '1':
                        x_pos += 1.0
                    else:
                        x_neg += 1.0
                    line = line.rstrip().split(' ')
                    y.append(float(line[0].strip()))
                    X.append([])
                    if not names:
                        for X_data in line[1:]:
                            X[-1].append(float(X_data.split(':')[1].strip()))
                            names.append(X_data.split(':')[0].strip())
                    else:
                        for X_data in line[1:]:
                            try:
                                X[-1].append(float(X_data.split(':')[1].strip()))
                            except:
                                X[-1].append(np.nan)
        num_features = len(line[1:])

        if impute_bool:
            incompFile = open(incompName, 'r')
            for line in incompFile:
                if line[0] != '#' and (line[0] == '1' or random.random() > 0.8):
                    if random.random() > 0.5:#remove data to see if results get worse
                        if line[0] == '1':
                            x_pos += 1.0
                        else:
                            x_neg += 1.0
                        line = line.rstrip().split(' ')
                        y_ic.append(float(line[0].strip()))
                        X_ic.append([])
                        if not names:
                            for X_data in line[1:]:
                                X_ic[-1].append(float(X_data.split(':')[1].strip()))
                                names_ic.append(X_data.split(':')[0].strip())
                        else:
                            for X_data in line[1:]:
                                try:
                                    X_ic[-1].append(float(X_data.split(':')[1].strip()))
                                except:
                                    X_ic[-1].append(np.nan)
            num_features_imp = len(line[1:])

        for line in investigationFile:
            if line[0] != '#':
                line = line.rstrip().split(' ')
                y_invest.append(float(line[0].strip()))
                X_invest.append([])
                if not names:
                    for X_data in line[1:]:
                        X_invest[-1].append(float(X_data.split(':')[1].strip()))
                        names_invest.append(X_data.split(':')[0].strip())
                else:
                    for X_data in line[1:]:
                        try:
                            X_invest[-1].append(float(X_data.split(':')[1].strip()))
                        except:
                            X_invest[-1].append(np.nan)

        X_full = np.array(X)

        # Create a dataframe with the feature variables
        df = pd.DataFrame(X_full, columns = names)
        #df_ic = pd.DataFrame(X_ic, columns = [''] + names)

        # View the top 5 rows
        print(df.head())
        #print(df_ic.head())

        # Heatmap for feature selection
        if args.heatmap:
            print('Calculating heatmap ')
            df_new = df.iloc
            df_new['Result'] = y
            corrmat = df_new.corr()
            top_corr_features = corrmat.index
            plt.figure(figsize=(50,50))
            #plot heat map
            g=sns.heatmap(df_new[top_corr_features].corr(),cmap="RdYlGn",xticklabels=1,yticklabels=1)
            plt.savefig('class/' + outDir + '/heatmap.png')
            plt.close()
            print('Done!')

        # split data into training and test set
        print('Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)#, random_state=0)

        # Impute missing values
        if impute_bool:
            gc.collect()
            print('Imputing...')
            imp = IterativeImputer(missing_values=np.nan, max_iter=300, estimator=BayesianRidge())
            imp.fit(X_train) #complete data
            X_ic = imp.transform(X_ic) #incomplete data
            X_invest = imp.transform(X_invest)
            X_ictrain, X_ictest, y_ictrain, y_ictest = train_test_split(X_ic, y_ic, test_size=0.2)#, random_state=0)
            X_ictrain = pd.DataFrame(X_ictrain, columns = names)
            X_ictest = pd.DataFrame(X_ictest, columns = names)
            y_ictrain = pd.DataFrame(y_ictrain, columns = ['label'])
            y_ictest = pd.DataFrame(y_ictest, columns = ['label'])
            y_train = pd.DataFrame(y_train, columns = ['label'])
            y_test = pd.DataFrame(y_test, columns = ['label'])

            X_train = X_train.append(X_ictrain)
            y_train = y_train.append(y_ictrain)
            X_test = X_test.append(X_ictest)
            y_test = y_test.append(y_ictest)
        else:
            y_train = pd.DataFrame(y_train, columns = ['label'])
            y_test = pd.DataFrame(y_test, columns = ['label'])

        # Perform feature selection with SelectKBest
        if num_features > 30:
            gc.collect()
            print('Performing feature selection...')
            if args.randomforest:
                classifier = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)#, random_state=0)
            elif args.extratrees:
                classifier = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)#, random_state=0)
            elif args.supportvector:#SVC does not work with classifier-based feature selection, so a different algo is necessary
                classifier = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)#, random_state=0)
            elif args.adaboost:
                classifier = AdaBoostClassifier(n_estimators=500)#, random_state=0)
            elif args.gradientboost:
                classifier = GradientBoostingClassifier(n_estimators=500, min_samples_split=4)#, random_state=0)
            if args.selectmodel:
                bestfeatures = SelectFromModel(estimator=classifier, threshold='0.5*mean')
            elif args.selectkbest:
                bestfeatures = SelectKBest(score_func=mutual_info_classif, k=30)
            elif args.boruta:
                bestfeatures = BorutaPy(classifier, n_estimators='auto')
            elif args.principal:
                bestfeatures = decomposition.PCA(n_components=20)
            fit = bestfeatures.fit(np.array(X_train),np.array(y_train).ravel())
            if args.boruta:
                bestfeatures.support_
                bestfeatures.ranking_
                green_area = X_train.columns[bestfeatures.support_].to_list()
                blue_area = X_train.columns[bestfeatures.support_weak_].to_list()
                print('features in the green area:', green_area)
                print('features in the blue area:', blue_area)
            X_train = bestfeatures.fit_transform(np.array(X_train), np.array(y_train).ravel())
            #num_features = bestfeatures.n_features_ §! See if this is now not necessary any more
            num_features = bestfeatures.transform(np.array(X_test)).shape[1]
            X_test = bestfeatures.transform(np.array(X_test))
            X_invest = bestfeatures.transform(np.array(X_invest))
            if args.selectkbest:
                dfscores = pd.DataFrame(fit.scores_)
                dfcolumns = pd.DataFrame(df.columns)
                featureScores = pd.concat([dfcolumns,dfscores],axis=1)
                featureScores.columns = ['Specs','Score']
                print(featureScores.nlargest(11,'Score'))
            print('Done!')
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            X_invest = pd.DataFrame(X_invest)
            if impute_bool:
                X_ictrain = bestfeatures.transform(np.array(X_ictrain))
                X_ictest = bestfeatures.transform(np.array(X_ictest))
                X_ictrain = pd.DataFrame(X_ictrain)
                X_ictest = pd.DataFrame(X_ictest)

        # Heatmap for feature selection
        if args.heatmap:
            print('Calculating heatmap ')
            df_new = X_train.iloc
            df_new['Result'] = y_train
            corrmat = df_new.corr()
            top_corr_features = corrmat.index
            plt.figure(figsize=(50,50))
            #plot heat map
            g=sns.heatmap(df_new[top_corr_features].corr(),cmap="RdYlGn",xticklabels=1,yticklabels=1)
            plt.savefig('class/' + outDir + '/heatmap2.png')
            plt.close()
            print('Done!')

        # reduce amount of negative training data
        y_container = []
        X_container = []
        for cys in range(len(y_train)):
            if (y_train.iloc[cys, 0] == 1.0 and random.random() < x_neg/x_pos) or (y_train.iloc[cys, 0] == 0.0 and random.random() < x_pos/x_neg):
                y_container.append(y_train.iloc[cys, 0])
                X_container.append(X_train.iloc[cys, :])
        y_train = y_container
        X_train = X_container

        # Parameter selection
        if args.gridsearch:
            gc.collect()
            print('Selecting parameters...')
            if args.randomforest or args.extratrees or args.gradientboost:
                estimators = []
                for estimator in range(3):
                    estimators.append((estimator+1)*500)
                m_feats = []
                for m_feat in range(3):#int(len(X_train[0])/20)):
                    m_feats.append((m_feat+2)*5)
                m_samples = []
                for m_sample in range(3):
                    m_samples.append((m_sample*2)+2)
                param_grid = {'n_estimators':estimators, 'max_features':m_feats, 'min_samples_split':m_samples}
            if args.randomforest:
                grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
            elif args.extratrees:
                grid_search = GridSearchCV(ExtraTreesClassifier(), param_grid)
            elif args.supportvector:
                Cs = []
                for C in range(10):
                    Cs.append(np.power(2.0, C-5.0))
                gammas = []
                for Gamma in range(15):
                    gammas.append(np.power(2.0, Gamma-15.0))
                param_grid = {'C': Cs, 'gamma' : gammas}
                grid_search = GridSearchCV(SVC(), param_grid)
            elif args.adaboost:
                estimators = []
                for estimator in range(5):
                    estimators.append((estimator+1)*250)
                learners = []
                for learner in range(10):
                    learners.append((learner+1)*0.2)
                param_grid = {'learning_rate': learners, 'n_estimators':estimators}
                grid_search = GridSearchCV(AdaBoostClassifier(), param_grid)
            elif args.gradientboost:
                grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid)
            grid_search = GridSearchCV(ExtraTreesClassifier(), param_grid)
            grid_search.fit(X_train, y_train)
            params = grid_search.best_params_
            print(params)
            resultFile.write('n_estimators: ' + '{:.0e}'.format(params['n_estimators']) + ' max_features: ' + '{:.0e}'.format(params['max_features']) + ' min_samples_split: ' + '{:.0e}'.format(params['min_samples_split']))
            print('Done!')

        if args.supportvector:
            params = {'C':1.0, 'gamma':0.008}
        else:
            if num_features > 20:
                params = {'n_estimators':500, 'max_features':20, 'min_samples_split':4}
            else:
                params = {'n_estimators':500, 'max_features':num_features, 'min_samples_split':4}

        # Feature Scaling
        print('Feature scaling...')
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_invest = sc.transform(X_invest)

        # Training
        gc.collect()
        print('Training...')
        if args.randomforest:
            classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'], min_samples_split=params['min_samples_split'])#, random_state=0)
        elif args.extratrees:
            classifier = ExtraTreesClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'], min_samples_split=params['min_samples_split'])#, random_state=0)
        elif args.supportvector:
            classifier = SVC(cache_size=2000, probability=True, C=params['C'], gamma=params['gamma'])#, random_state=0)
        elif args.adaboost:
            classifier = AdaBoostClassifier(n_estimators=500)#, random_state=0)
        elif args.gradientboost:
            classifier = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'], min_samples_split=params['min_samples_split'])#, random_state=0)
        classifier.fit(X_train, y_train)
        classifier_roc = metrics.plot_roc_curve(classifier, X_test, y_test, color='darkorange')
        interp_tpr = interp(mean_fpr, classifier_roc.fpr, classifier_roc.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(classifier_roc.roc_auc)
        resultFile.write(' AUC Test: ' + str(classifier_roc.roc_auc))
        y_pred = classifier.predict(X_test)
        y_pred_invest = classifier.predict(X_invest)

        # Case study
        y_test = y_test['label'].values.tolist()
        X_new = np.append(X_train, X_test, axis = 0)
        y_new = np.append(y_train, y_test, axis = 0)
        classifier.fit(X_new, y_new)
        classifier_invest = metrics.plot_roc_curve(classifier, X_invest, y_invest, color='darkorange')
        plt.close()
        invest_interp_tpr = interp(invest_mean_fpr, classifier_invest.fpr, classifier_invest.tpr)
        invest_interp_tpr[0] = 0.0
        invest_tprs.append(invest_interp_tpr)
        invest_aucs.append(classifier_invest.roc_auc)
        resultFile.write(' AUC Case: ' + str(classifier_invest.roc_auc) + '\n')

        # Evaluation
        print('Probabilities:')
        print(classifier.predict_proba(X_test)[0:10])
        print('Predicted results:')
        print(y_pred[0:10])
        print('Actual results:')
        print(y_test[0:10])
        print('Accuracy:')
        print(metrics.accuracy_score(y_test, y_pred))
        if args.randomforest or args.extratrees or args.adaboost or args.gradientboost:
            print('Feature importance:')
            feature_imp = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
            print(feature_imp)
            plt.clf()
            feature_imp.nlargest(10).plot(kind='barh')
            #plt.show()
            plt.clf()
            print(list(zip(X_train, classifier.feature_importances_))[0])
            print(classifier.feature_importances_)
        print('Confusion matrix:')
        True_Pos = len([i for i, j in zip(y_pred, y_test) if (i == j and i == 1)])
        True_Neg = len([i for i, j in zip(y_pred, y_test) if (i == j and i == 0)])
        False_Pos = len([i for i, j in zip(y_pred, y_test) if (i != j and i == 1)])
        False_Neg = len([i for i, j in zip(y_pred, y_test) if (i != j and i == 0)])
        print('TP: ' + str(True_Pos))
        print('TN: ' + str(True_Neg))
        print('FP: ' + str(False_Pos))
        print('FN: ' + str(False_Neg))
        TP_sum += True_Pos
        TN_sum += True_Neg
        FP_sum += False_Pos
        FN_sum += False_Neg

    try:
        outFile.write('PPV: ' + str(TP_sum/(TP_sum+FP_sum)) + ' NPV: ' + str(TN_sum/(TN_sum+FN_sum)) +
        ' Sensitivity: ' + str(TP_sum/(TP_sum+FN_sum)) + ' Specificity: ' + str(TN_sum/(TN_sum+FP_sum)) + '\n')
    except ZeroDivisionError:
        outFile.write('ZeroDivisionError')
    #print(pd.crosstab(y_pred, y_test, rownames=['Actual Redox'], colnames=['Predicted Redox']))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Results for Complex III
    print('Results for tested protein:\n')
    print('Probabilities:')
    print(classifier.predict_proba(X_invest)[0:5])
    print('Predicted results:')
    print(y_pred_invest)
    all_preds.append(y_pred_invest)
    print('Actual results:')
    print(np.array(y_invest))
    print('Accuracy:')
    for resultcount in range(len(classifier.predict_proba(X_invest))):
        predFile.write(str(classifier.predict_proba(X_invest)[resultcount]) + ' ' + str(y_pred_invest[resultcount]) + ' ' + str(y_invest[resultcount]) + '\n')
    print(metrics.accuracy_score(y_invest, y_pred_invest))
    predFile.write('Accuracy: ' + str(metrics.accuracy_score(y_invest, y_pred_invest)))

    # Creating a bar plot for feature importance
    if args.randomforest or args.extratrees or args.adaboost or args.gradientboost:
        sns.barplot(x=feature_imp[0:14], y=feature_imp.index[0:14])
        plt.clf()#§!Rest of feature imp
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Important Features for Redox Cysteine Prediction")
        plt.legend()
        plt.savefig('class/' + outDir + '/feature_importance_summary' + str(count) + '.png')
        #plt.show() # If you want to see the feature importance
        plt.clf()

    # Save the trees
    # Pull out one tree from the forest
    #tree = classifier.estimators_[5]

    # Export the image to a dot file
    #export_graphviz(tree, out_file = 'tree.dot', rounded = True, precision = 1)

    # Use dot file to create a graph
    #(graph, ) = pydot.graph_from_dot_file('tree.dot')

    # Write graph to a png file
    #graph.write_png('full/flava/tree' + str(count) + '.png')

    # Limit depth of tree to 3 levels
    #clf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)
    #clf_small.fit(X_train, y_train)

    # Extract the small tree
    #tree_small = clf_small.estimators_[5]

    # Save the tree as a png image
    #export_graphviz(tree_small, out_file = 'small_tree.dot', rounded = True, precision = 1)

    #(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

    #graph.write_png('full/flava/small_tree' + str(count) + '.png');
    #gc.collect()


for invest in range(len(all_preds)-1):
    for invest_cyst in range(len(all_preds[0])):
        all_preds[0][invest_cyst] += all_preds[invest + 1][invest_cyst]
print(np.array(all_preds[0]))
Final_True_Pos = 0
Final_False_Pos = 0
Final_True_Neg = 0
Final_False_Neg = 0
for i in range(len(all_preds[0])):
    if all_preds[0][i] >= 6 and np.array(y_invest)[i] == 1:
        Final_True_Pos += 1
    if all_preds[0][i] >= 6 and np.array(y_invest)[i] == 0:
        Final_False_Pos += 1
    if all_preds[0][i] < 6 and np.array(y_invest)[i] == 0:
        Final_True_Neg += 1
    if all_preds[0][i] < 6 and np.array(y_invest)[i] == 1:
        Final_False_Neg += 1
resultFile.write('True Positives: ' + str(Final_True_Pos) + ', False Positives: ' + str(Final_False_Pos) + ', True Negatives: ' + str(Final_True_Neg) + ', False Negatives: ' + str(Final_False_Neg) + '\n')
resultFile.write('Predicted results|Actual results\n')
for i in range(len(np.array(all_preds[0]))):
    resultFile.write(str(np.array(all_preds[0])[i]) + ' | ' + str(np.array(y_invest)[i]) + '\n')

# ROC with cross-validation
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right")
plt.savefig('class/' + outDir + '/roc.png')
plt.close()

# ROC with cross-validation for case study
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

invest_mean_tpr = np.mean(invest_tprs, axis=0)
invest_mean_tpr[-1] = 1.0
invest_mean_auc = metrics.auc(invest_mean_fpr, invest_mean_tpr)
invest_std_auc = np.std(invest_aucs)
ax.plot(invest_mean_fpr, invest_mean_tpr, color='b',
        label=r'Mean ROC case(AUC = %0.2f $\pm$ %0.2f)' % (invest_mean_auc, invest_std_auc),
        lw=2, alpha=.8)

invest_std_tpr = np.std(invest_tprs, axis=0)
invest_tprs_upper = np.minimum(invest_mean_tpr + invest_std_tpr, 1)
invest_tprs_lower = np.maximum(invest_mean_tpr - invest_std_tpr, 0)
ax.fill_between(invest_mean_fpr, invest_tprs_lower, invest_tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right")
plt.savefig('class/' + outDir + '/roc_pred.png')
plt.close()
